import numpy as np
from ovito.io import *
from ovito.modifiers import *
from ovito.pipeline import *
import numpy as np
import subprocess
import random
import os

class TDE_simulation:

    def __init__(self, atom_id, direction_file=None, num_thermalisations=1,
                 lammps_data_file='YBCO.data', lammps_input_file='YBCO.in',
                 temperature=25):


        self.direction_file = direction_file
        if self.direction_file == None:
            # Here we can implement some logic to do random directions
            exit()
        else:
            try:
                with open(self.direction_file, 'r') as f:
                    # Direction data file contains 3 entries on each row corresponding to vector direction.
                    self.direction_data = f.readlines()
                    data = []
                    for line in self.direction_data:
                        x, y, z = line.split()
                        data.append([float(x), float(y), float(z)])
                    self.direction_data = data

            except FileNotFoundError:
                print(f"Specified direction file: {direction_file} not found, exiting...")
                exit()

        # Number of times to restart simulations from different starting equilibration.
        self.num_thermalisations = num_thermalisations

        # ID of PKA atom
        self.atom_id = atom_id

        # File containing lammps position data
        self.lammps_data_file = lammps_data_file
        self.temperature = temperature

        # Initialise PKA mass at start of calcualtions
        self.get_pka_mass()

        self.run_line = 'mpirun -np 64 /storage/hpc/51/dickson5/codes/tablammps/lammps_w_hdf5/build/lmp -in'


    def get_pka_mass (self):

        # Pass lammps data file to retrieve PKA info
        with open (self.lammps_data_file, 'r') as f:
            lines = f.readlines()
        for index, line in enumerate(lines):
            if 'Atoms' in line:
                atom_data_start = index + 2
        data_lines = lines[atom_data_start:]

        # Parse data for PKA atom
        for line in data_lines:
            atom_data = line.split()
            if int(atom_data[0]) == int(self.atom_id):
                atom_type = atom_data[1]
                break
        mass_dict = {
            1 : 15.999,
            2 : 63.456,
            3 : 88.90585,
            4 : 137.327
        }
        self.pka_mass =  mass_dict[int(atom_type)]

    def restart_check (self):

        files = os.listdir()
        thermal_files = [file for file in files if file.startswith('thermalisation')]
        thermal_nums = [int(file.split('_')[1]) for file in thermal_files]
        thermal_start_num =  int(np.max(thermal_nums))
        print(thermal_start_num)

        try:
            with open (f'thermalisation_{thermal_start_num}/data.txt', 'r') as f:
                lines = f.readlines()
            direction_start_num = int(lines[-1].split()[0])
        except FileNotFoundError:
            direction_start_num = 0

        return direction_start_num, thermal_start_num



    def calculate_pka_vel (self, pka_vector, pka_energy):

        # PKA vector is list of floats
        eV_to_J = 1.60218e-19  # J/eV
        amu_to_kg = 1.66e-27   # kg/amu

        pka_vector_magnitude = np.linalg.norm(pka_vector)
        unit_vector = pka_vector / pka_vector_magnitude


        pka_velocity = np.sqrt(2 * pka_energy * eV_to_J / (self.pka_mass * amu_to_kg))
        pka_velocity = pka_velocity/100 #in lammps units

        pka_velocity_vector = pka_velocity * unit_vector

        print(f"Velocity magnitude: {pka_velocity:.2f} m/s")
        print(f"Velocity vector: {pka_velocity_vector}")

        # pka velocity vector is list of floats
        return pka_velocity_vector

    def defect_check (self):

        print('Checking for defects..')
        filename = 'prod.data'
        pipeline = import_file(filename)


        ws_modifier = WignerSeitzAnalysisModifier(
            per_type_occupancies = True
            #eliminate_cell_deformation = True,
            #affine_mapping = ReferenceConfigurationModifier.AffineMapping.ToReference
        )


        pipeline.modifiers.append(ws_modifier)
        for frame in range(1, pipeline.source.num_frames):
            data = pipeline.compute(frame)
            occupancies = data.particles['Occupancy'].array
            occupancy2 = 0 #total num interstitial
            occupancy0 = 0 #total num vacancies
            # Get the site types as additional input:
            site_type = data.particles['Particle Type'].array
            # Calculate total occupancy of every site:
            total_occupancy = np.sum(occupancies, axis=1)
            #print(total_occupancy)
            for element in total_occupancy:
                if element == 0:
                    occupancy0 +=1
                if element >= 2:
                    occupancy2 += (1 + (element-2))
            # Set up a particle selection by creating the Selection property:
            selection = data.particles_.create_property('Selection')

            # Select A-sites occupied by exactly one B, C, or D atom
            # (the occupancy of the corresponding atom type must be 1, and all others 0)
            selection[...] =((site_type == 1) & (occupancies[:, 1] == 1) & (total_occupancy == 1)) | \
                            ((site_type == 1) & (occupancies[:, 2] == 1) & (total_occupancy == 1)) | \
                            ((site_type == 1) & (occupancies[:, 3] == 1) & (total_occupancy == 1)) | \
                            ((site_type == 2) & (occupancies[:, 2] == 1) & (total_occupancy == 1)) | \
                            ((site_type == 2) & (occupancies[:, 3] == 1) & (total_occupancy == 1)) | \
                            ((site_type == 4) & (occupancies[:, 2] == 1) & (total_occupancy == 1))
            antisite_indices = np.where(selection == 1)[0]


            # Count the total number of antisite defects
            antisite_count = np.count_nonzero(selection[...])

            # Output the total number of antisites as a global attribute:
            data.attributes['Antisite_count'] = antisite_count
            tot_num_defects = antisite_count + occupancy0 + occupancy2

            defect_count = tot_num_defects
        return defect_count

    def modify_lammps_in (self, filename, pka_vector, seed):

        with open (filename, 'r') as f:
            lines = f.readlines()

        with open (filename, 'w') as f:
            for line in lines:
                if 'MYT' in line:
                    f.write(f'variable T equal {self.temperature}\n')
                elif 'MYPKAVEL' in line:
                    f.write(f'velocity PKA set {pka_vector[0]} {pka_vector[1]} {pka_vector[2]}\n')
                elif 'MYPKAID' in line:
                    f.write(f'group PKA id {self.atom_id}\n')
                elif 'MYSEED' in line:
                    f.write(f'velocity all create ${{T}} {seed}\n')
                else:
                    f.write(line)
    def initialise_equil(self):

        with open ('equil.in', 'r') as f:
            lines = f.readlines()

        with open ('equil.in', 'w') as f:
            for line in lines:
                if 'variable T equal' in line:
                    f.write(f'variable T equal {self.temperature}\n')
                elif 'velocity all create' in line:
                    f.write(f'velocity all create ${{T}} 1234\n')
                else:
                    f.write(line)



    def run_tde_loop (self):

        # Logic for restarts
        if os.path.exists('thermalisation_1'):
            direction_start, thermal_start = self.restart_check()
        else:
            direction_start = 0
            thermal_start = 1
            self.initialise_equil()
        # This changes temp to desired temperature for equilibration.
        self.modify_lammps_in(filename = 'equil.in', pka_vector = None, seed = None)
        # Run equilibration
        subprocess.run(self.run_line + ' equil.in', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # We now have YBCO.data as an equilibriated structure

        # Loop over thermalisations
        for i in range ((thermal_start-1),self.num_thermalisations):
            print(f'Starting thermalisation loop {i+1}...')
            try:
                os.mkdir(f'thermalisation_{i+1}')
            except FileExistsError:
                pass
            # Generate random seed for another short equilibration
            random_seed = random.randint(1000, 9999)
            # Run data will contain all info for TDE of each direction
            run_data = []

            if os.path.exists(f'thermalisation_{i+1}/data.txt'):
                with open (f'thermalisation_{i+1}/data.txt', 'r') as f:
                    for line in f.readlines():
                        run_data.append(line.split())

            for index, direction in enumerate(self.direction_data[direction_start:], start=direction_start):
                print('direction_start', direction_start)
                # If data.txt exists, we overwrite for every new completed direction
                if len(run_data) > 0:
                    with open(f'thermalisation_{i+1}/data.txt', 'w') as f:
                        for line in run_data:
                            f.write(f'{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n')

                try:
                    os.mkdir(f'thermalisation_{i+1}/direction_{index+1}')
                except FileExistsError:
                    pass
                os.chdir(f'thermalisation_{i+1}/direction_{index+1}')


                # start TDE sim from 1 eV
                for energy in range (1, 100, 1):
                    os.system('cp ../../tde.in .')
                    # Calculate PKA velocity for given energy
                    vel_vector = self.calculate_pka_vel(direction, pka_energy = energy)
                    # Change direction and seed in tde.in
                    self.modify_lammps_in(filename = 'tde.in', pka_vector = vel_vector, seed = random_seed)
                    # Run tde.in
                    subprocess.run(self.run_line + ' tde.in', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    # Check for displacements
                    defect_count = self.defect_check()
                    if defect_count == 0:
                        print(f'No defects found for direction {index+1}, thermalisation {i+1}, energy {energy}')
                    else:
                        print((f'Defect found for direction {index+1}, thermalisation {i+1}, energy {energy}'))
                        run_data.append([index+1, energy, direction[0], direction[1], direction[2]])
                        break
                os.chdir('../..')
                with open(f'thermalisation_{i+1}/data.txt', 'w') as f:
                        for line in run_data:
                            f.write(f'{line[0]} {line[1]} {line[2]} {line[3]} {line[4]}\n')
            direction_start = 0





TDE_simulation(atom_id = 540, direction_file='directions25K.txt', num_thermalisations=50, temperature=360).run_tde_loop()
