# Problem Set 4: Simulating the Spread of Disease and Bacteria Population Dynamics
# Name:
# Collaborators (Discussion):
# Time:

import math
import numpy as np
import pylab as pl
import random


##########################
# End helper code
##########################

class NoChildException(Exception):
    """
    NoChildException is raised by the reproduce() method in the SimpleBacteria
    and ResistantBacteria classes to indicate that a bacteria cell does not
    reproduce. You should use NoChildException as is; you do not need to
    modify it or add any code.
    """


def make_one_curve_plot(x_coords, y_coords, x_label, y_label, title):
    """
    Makes a plot of the x coordinates and the y coordinates with the labels
    and title provided.

    Args:
        x_coords (list of floats): x coordinates to graph
        y_coords (list of floats): y coordinates to graph
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): title for the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords)
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


def make_two_curve_plot(x_coords,
                        y_coords1,
                        y_coords2,
                        y_name1,
                        y_name2,
                        x_label,
                        y_label,
                        title):
    """
    Makes a plot with two curves on it, based on the x coordinates with each of
    the set of y coordinates provided.

    Args:
        x_coords (list of floats): the x coordinates to graph
        y_coords1 (list of floats): the first set of y coordinates to graph
        y_coords2 (list of floats): the second set of y-coordinates to graph
        y_name1 (str): name describing the first y-coordinates line
        y_name2 (str): name describing the second y-coordinates line
        x_label (str): label for the x-axis
        y_label (str): label for the y-axis
        title (str): the title of the graph
    """
    pl.figure()
    pl.plot(x_coords, y_coords1, label=y_name1)
    pl.plot(x_coords, y_coords2, label=y_name2)
    pl.legend()
    pl.xlabel(x_label)
    pl.ylabel(y_label)
    pl.title(title)
    pl.show()


##########################
# PROBLEM 1
##########################

class SimpleBacteria(object):
    """A simple bacteria cell with no antibiotic resistance"""

    def __init__(self, birth_prob, death_prob):
        """
        Args:
            birth_prob (float in [0, 1]): Maximum possible reproduction
                probability
            death_prob (float in [0, 1]): Maximum death probability
        """
        self.birth_prob = birth_prob
        self.death_prob = death_prob
        #pass  # TODO

    def is_killed(self):
        """
        Stochastically determines whether this bacteria cell is killed in
        the patient's body at a time step, i.e. the bacteria cell dies with
        some probability equal to the death probability each time step.

        Returns:
            bool: True with probability self.death_prob, False otherwise.
        """
        if random.random() < self.death_prob:
            #print("bacteria is killed")
            return True
        else:
            #print ('bacteria survived')
            return False
        #pass  # TODO

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the Patient and
        TreatedPatient classes.

        The bacteria cell reproduces with probability
        self.birth_prob * (1 - pop_density).

        If this bacteria cell reproduces, then reproduce() creates and returns
        the instance of the offspring SimpleBacteria (which has the same
        birth_prob and death_prob values as its parent).

        Args:
            pop_density (float): The population density, defined as the
                current bacteria population divided by the maximum population

        Returns:
            SimpleBacteria: A new instance representing the offspring of
                this bacteria cell (if the bacteria reproduces). The child
                should have the same birth_prob and death_prob values as
                this bacteria.

        Raises:
            NoChildException if this bacteria cell does not reproduce.
        """
        #print ("Bacteria birth rate", self.birth_prob , "pop dentity", pop_density)
        bacteria_reproduce = self.birth_prob * (1 - pop_density)
        #print("bacteria repoduce rate", bacteria_reproduce, random.random())
        if random.random() < bacteria_reproduce:
            #print("Bacteria reproduce", SimpleBacteria(self.birth_prob, self.death_prob))
#            print ("bacteria_reproduce and return bacteria", SimpleBacteria(self.birth_prob, self.death_prob))
            offspring_bacteria =  SimpleBacteria(self.birth_prob, self.death_prob)
            return offspring_bacteria
        raise NoChildException()
        #pass  # TODO

class Patient(object):
    """
    Representation of a simplified patient. The patient does not take any
    antibiotics and his/her bacteria populations have no antibiotic resistance.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria (list of SimpleBacteria): The bacteria in the population
            max_pop (int): Maximum possible bacteria population size for
                this patient
        """
        self.bacteria = bacteria
        self.max_pop = max_pop
        pass  # TODO

    def get_total_pop(self):
        """
        Gets the size of the current total bacteria population.

        Returns:
            int: The total bacteria population
        """
        return len(self.bacteria)
        pass  # TODO

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute the following steps in
        this order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. Calculate the current population density by dividing the surviving
           bacteria population by the maximum population. This population
           density value is used for the following steps until the next call
           to update()

        3. Based on the population density, determine whether each surviving
           bacteria cell should reproduce and add offspring bacteria cells to
           a list of bacteria in this patient. New offspring do not reproduce.

        4. Reassign the patient's bacteria list to be the list of surviving
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """

        new_list= []
        offspring = []
        suv_bacteria = 0
        pop = 0
        "determine if each bacteria live or die and create new list"
        #print (self.bacteria)
        for item in self.bacteria:
#print (item, item.is_killed(), new_list)
            if not item.is_killed():
                new_list.append(item)
#            print("bacteria list", new_list)
        "Calculate the population denist by adding all survived bacteria"
        pop_den =  len(new_list) / self.max_pop
        #print(pop_den)
        "define if the bacteria in the new list will reproduce"
        for item in new_list:
            #print ("item", new_list, item.reproduce(pop_den))
            try: #item.reproduce(pop_den):
                #print ("Bacteria reproduce, Offspring", item.reproduce(pop_den))
                offspring.append(item.reproduce(pop_den))
            except:
                #print("no offspring")
                pass
        #print("Bacteria list", new_list,"offspring", offspring)
        self.bacteria.clear()
        self.bacteria = new_list + offspring
#        print (self.bacteria)
        return  len(self.bacteria)
        #pass  # TODO
"""
Bacteria_test = SimpleBacteria(1, 0)
killed=0
survived=0
for i in range (100):
    if Bacteria_test.is_killed():
        killed+=1
    if not Bacteria_test.is_killed():
        survived+=1
print("killed", killed, "\nsurvived", survived)
#print("bacteria reproduce", Bacteria_test.reproduce(0.2))
bacteria = [Bacteria_test]
max_pop = 1
test_patient = Patient(bacteria,max_pop)
for i in range(10):
#    print(bacteria,test_patient.get_total_pop())
    print ("list of bacteria", len(bacteria), "max pop", max_pop)
    max_pop = test_patient.update()
#    test_patient = Patient(bacteria,max_pop)
    print("Bacteria pop after",i,"trial", max_pop)
"""
##########################
# PROBLEM 2
##########################

def calc_pop_avg(populations, n):
    """
    Finds the average bacteria population size across trials at time step n

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j

    Returns:
        float: The average bacteria population size at time step n
    """
    num_bacteria = 0
    population = 0
    while population < len(populations):
        #print (populations[population][n])
        num_bacteria = num_bacteria + populations[population][n]
        population +=1
    return (num_bacteria/len(populations))
    pass  # TODO


def simulation_without_antibiotic(num_bacteria,
                                  max_pop,
                                  birth_prob,
                                  death_prob,
                                  num_trials):
    """
    Run the simulation and plot the graph for problem 2. No antibiotics
    are used, and bacteria do not have any antibiotic resistance.

    For each of num_trials trials:
        * instantiate a list of SimpleBacteria
        * instantiate a Patient using the list of SimpleBacteria
        * simulate changes to the bacteria population for 300 timesteps,
          recording the bacteria population after each time step. Note
          that the first time step should contain the starting number of
          bacteria in the patient

    Then, plot the average bacteria population size (y-axis) as a function of
    elapsed time steps (x-axis) You might find the make_one_curve_plot
    function useful.

    Args:
        num_bacteria (int): number of SimpleBacteria to create for patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float in [0, 1]): maximum reproduction
            probability
        death_prob (float in [0, 1]): maximum death probability
        num_trials (int): number of simulation runs to execute

    Returns:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria in trial i at time step j
    """
    timesteps = 300
    bacteria =[]
    patient = Patient(bacteria, max_pop)
    populations =[]

    """ run the number of trials"""
    for p in range(num_trials):
        """ instantiate a list of SimpleBacteria """
        bacteria =[]
        for i in range(num_bacteria):
            bacteria.append(SimpleBacteria (birth_prob, death_prob))
        """instantiate a Patient using the list of SimpleBacteria"""
        patient = Patient(bacteria, max_pop)
        patient_time_steps = []
        """ simulate change for 300 timesteps"""
        patient_time_steps.append(patient.get_total_pop())
        for i in range(timesteps):
            #print ("timesteps:", i)
            patient_time_steps.append(patient.update())
            #print ("patient list at time steps",i,"Number of trials", p, patient_time_steps)
        populations.insert(p,patient_time_steps)
    #print("trial 0",populations[0],"\n", populations[0][1], "\ntrial 1", populations[1],"\n",populations[1][3])

    trial =0
    x_coords = []
    y_coords = []
    for i in range(timesteps):
        x_coords.append(i)
        y_coords.append(calc_pop_avg(populations, i))

    plt = make_one_curve_plot(x_coords, y_coords, x_label = "timesteps", y_label = "average bacteria", title = "Pb 2: simulation without antibiotic")


    return populations
    pass  # TODO
#random.seed(0)
#print ("simulaion", simulation_without_antibiotic(100, 1000, 0.1, 0.025, 50))
# When you are ready to run the simulation, uncomment the next line
#populations = simulation_without_antibiotic(100, 1000, 0.1, 0.025, 50)

##########################
# PROBLEM 3
##########################

def calc_pop_std(populations, t):
    """
    Finds the standard deviation of populations across different trials
    at time step t by:
        * calculating the average population at time step t
        * compute average squared distance of the data points from the average
          and take its square root

    You may not use third-party functions that calculate standard deviation,
    such as numpy.std. Other built-in or third-party functions that do not
    calculate standard deviation may be used.

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        float: the standard deviation of populations across different trials at
             a specific time step
    """
    average_population = calc_pop_avg(populations ,t)
    tot = 0.0
    for population in populations:
        tot += (population[t] - average_population) ** 2
    std = (tot/len(populations))**0.5
    return std
    pass  # TODO

#print (calc_pop_std(populations,0),calc_pop_std(populations,1) )

def calc_95_ci(populations, t):
    """
    Finds a 95% confidence interval around the average bacteria population
    at time t by:
        * computing the mean and standard deviation of the sample
        * using the standard deviation of the sample to estimate the
          standard error of the mean (SEM)
        * using the SEM to construct confidence intervals around the
          sample mean

    Args:
        populations (list of lists or 2D array): populations[i][j] is the
            number of bacteria present in trial i at time step j
        t (int): time step

    Returns:
        mean (float): the sample mean
        width (float): 1.96 * SEM

        I.e., you should return a tuple containing (mean, width)
    """
    means =0
    std =0
    samplesize = 15
    bad_estimate = 0
    num_trials = 50
    """ average of the bacteria population"""
    means = calc_pop_avg (populations, t)
    """ average, stadnard deviation and standard error of sample """
    for i in range (num_trials):
        sample = random.sample(populations, samplesize)
        #print (len(sample))
        sample_means = calc_pop_avg(sample,t)
        sample_std = calc_pop_std (sample,t)
        se = sample_std / (samplesize**0.5)
        width = 1.96*se
        if abs (means - sample_means ) > 1.96*se:
            bad_estimate+=1
    print ("Mean of sample:", sample_means,"\nstandard deviation of sample:", sample_std,"\nestimate Standard error (SEM):",se)
    print ("fraction outside 95%confidence interval=", bad_estimate / num_trials )
    return (sample_means,width )
    pass  # TODO

#for i in [0, 1, 10, 100, 200, 300]:
#calc_95_ci(populations, i)

##########################
# PROBLEM 4
##########################

class ResistantBacteria(SimpleBacteria):
    """A bacteria cell that can have antibiotic resistance."""

    def __init__(self, birth_prob, death_prob, resistant, mut_prob):
        """
        Args:
            birth_prob (float in [0, 1]): reproduction probability
            death_prob (float in [0, 1]): death probability
            resistant (bool): whether this bacteria has antibiotic resistance
            mut_prob (float): mutation probability for this
                bacteria cell. This is the maximum probability of the
                offspring acquiring antibiotic resistance
        """
        SimpleBacteria.__init__(self, birth_prob, death_prob)
        self.resistant = resistant
        self.mut_prob = mut_prob
        pass  # TODO

    def get_resistant(self):
        """Returns whether the bacteria has antibiotic resistance"""
        return random.choice([True, False])
        pass  # TODO

    def is_killed(self):
        """Stochastically determines whether this bacteria cell is killed in
        the patient's body at a given time step.

        Checks whether the bacteria has antibiotic resistance. If resistant,
        the bacteria dies with the regular death probability. If not resistant,
        the bacteria dies with the regular death probability / 4.

        Returns:
            bool: True if the bacteria dies with the appropriate probability
                and False otherwise.
        """
        if self.resistant:
            if random.random() < self.death_prob:
                #print("bacteria is resistant and killed")
                return True
            else:
                #print ('bacteria is resistant and survived')
                return False
        else:
            if random.random() < (self.death_prob/4):
                #print("bacteria is killed")
                return True
            else:
                #print ('bacteria survived')
                return False
        pass  # TODO

    def reproduce(self, pop_density):
        """
        Stochastically determines whether this bacteria cell reproduces at a
        time step. Called by the update() method in the TreatedPatient class.

        A surviving bacteria cell will reproduce with probability:
        self.birth_prob * (1 - pop_density).

        If the bacteria cell reproduces, then reproduce() creates and returns
        an instance of the offspring ResistantBacteria, which will have the
        same birth_prob, death_prob, and mut_prob values as its parent.

        If the bacteria has antibiotic resistance, the offspring will also be
        resistant. If the bacteria does not have antibiotic resistance, its
        offspring have a probability of self.mut_prob * (1-pop_density) of
        developing that resistance trait. That is, bacteria in less densely
        populated environments have a greater chance of mutating to have
        antibiotic resistance.

        Args:
            pop_density (float): the population density

        Returns:
            ResistantBacteria: an instance representing the offspring of
            this bacteria cell (if the bacteria reproduces). The child should
            have the same birth_prob, death_prob values and mut_prob
            as this bacteria. Otherwise, raises a NoChildException if this
            bacteria cell does not reproduce.
        """
        #print ("Bacteria birth rate", self.birth_prob , "pop dentity", pop_density)
        bacteria_reproduce = self.birth_prob * (1 - pop_density)
        #print("bacteria repoduce rate", bacteria_reproduce, random.random())
        if random.random() < bacteria_reproduce:
            #print("Bacteria reproduce", SimpleBacteria(self.birth_prob, self.death_prob))
        #            print ("bacteria_reproduce and return bacteria", SimpleBacteria(self.birth_prob, self.death_prob))
            if self.resistant:
                offspring_bacteria =  ResistantBacteria(self.birth_prob, self.death_prob, True ,self.mut_prob)
            else:
                """ calculate the prob of mutation"""
                off_rest_prob = self.mut_prob * (1-pop_density)
                if random.random() < off_rest_prob:
                    offspring_bacteria =  ResistantBacteria(self.birth_prob, self.death_prob, True , self.mut_prob)
                else:
                    offspring_bacteria =  ResistantBacteria(self.birth_prob, self.death_prob, False , self.mut_prob)
            return offspring_bacteria
        raise NoChildException()

        pass  # TODO


class TreatedPatient(Patient):
    """
    Representation of a treated patient. The patient is able to take an
    antibiotic and his/her bacteria population can acquire antibiotic
    resistance. The patient cannot go off an antibiotic once on it.
    """
    def __init__(self, bacteria, max_pop):
        """
        Args:
            bacteria: The list representing the bacteria population (a list of
                      bacteria instances)
            max_pop: The maximum bacteria population for this patient (int)

        This function should initialize self.on_antibiotic, which represents
        whether a patient has been given an antibiotic. Initially, the
        patient has not been given an antibiotic.

        Don't forget to call Patient's __init__ method at the start of this
        method.
        """
        #print (self,bacteria, max_pop)
        Patient.__init__(self, bacteria, max_pop)
        self.on_antibiotic = False
        pass  # TODO

    def set_on_antibiotic(self):
        """
        Administer an antibiotic to this patient. The antibiotic acts on the
        bacteria population for all subsequent time steps.
        """
        self.on_antibiotic = True
        return self.on_antibiotic
        pass  # TODO

    def get_resist_pop(self):
        """
        Get the population size of bacteria cells with antibiotic resistance

        Returns:
            int: the number of bacteria with antibiotic resistance
        """
        resistant_pop = 0
        #print( "extract qty of the resistant populaiotn ")
        for bac in self.bacteria:
            #print ("is bac resistant", bac.resistant)
            if bac.resistant:
                resistant_pop +=1
        return resistant_pop
        pass  # TODO

    def update(self):
        """
        Update the state of the bacteria population in this patient for a
        single time step. update() should execute these actions in order:

        1. Determine whether each bacteria cell dies (according to the
           is_killed method) and create a new list of surviving bacteria cells.

        2. If the patient is on antibiotics, the surviving bacteria cells from
           (1) only survive further if they are resistant. If the patient is
           not on the antibiotic, keep all surviving bacteria cells from (1)

        3. Calculate the current population density. This value is used until
           the next call to update(). Use the same calculation as in Patient

        4. Based on this value of population density, determine whether each
           surviving bacteria cell should reproduce and add offspring bacteria
           cells to the list of bacteria in this patient.

        5. Reassign the patient's bacteria list to be the list of survived
           bacteria and new offspring bacteria

        Returns:
            int: The total bacteria population at the end of the update
        """
        new_list= []
        resist_list = []
        offspring = []
        suv_bacteria = 0
        pop = 0
        "determine if each bacteria live or die and create new list"
        #print (self.bacteria)
        for item in self.bacteria:
        #print (item, item.is_killed(), new_list)
            if not item.is_killed():
                #print("bacteria is not kill and add a bacteria to the list with len of", len(new_list))
                new_list.append(item)
                #print ("len of light after added surviving bacteria", len(new_list))
        #            print("bacteria list", new_list)
        " test if the patient is on antibiotic and revise the list with resistant bacterias"
        if self.on_antibiotic:
            for bac in new_list:
                if bac.resistant:
                    resist_list.append(bac)
            new_list = resist_list

        "Calculate the population denist by adding all survived bacteria"
        pop_den =  len(new_list) / self.max_pop
        #print(pop_den)
        "define if the bacteria in the new list will reproduce"
        for item in new_list:
            #print ("item", new_list, item.reproduce(pop_den))
            try: #item.reproduce(pop_den):
                #print ("Bacteria reproduce, Offspring", item.reproduce(pop_den))
                offspring.append(item.reproduce(pop_den))
            except:
                #print("no offspring")
                pass
        #print("Bacteria list", new_list,"offspring", offspring)
        self.bacteria.clear()
        self.bacteria = new_list + offspring
        #        print (self.bacteria)
        return  len(self.bacteria)



##########################
# PROBLEM 5
##########################

def simulation_with_antibiotic(num_bacteria,
                               max_pop,
                               birth_prob,
                               death_prob,
                               resistant,
                               mut_prob,
                               num_trials):
    """
    Runs simulations and plots graphs for problem 4.

    For each of num_trials trials:
        * instantiate a list of ResistantBacteria
        * instantiate a patient
        * run a simulation for 150 timesteps, add the antibiotic, and run the
          simulation for an additional 250 timesteps, recording the total
          bacteria population and the resistance bacteria population after
          each time step

    Plot the average bacteria population size for both the total bacteria
    population and the antibiotic-resistant bacteria population (y-axis) as a
    function of elapsed time steps (x-axis) on the same plot. You might find
    the helper function make_two_curve_plot helpful

    Args:
        num_bacteria (int): number of ResistantBacteria to create for
            the patient
        max_pop (int): maximum bacteria population for patient
        birth_prob (float int [0-1]): reproduction probability
        death_prob (float in [0, 1]): probability of a bacteria cell dying
        resistant (bool): whether the bacteria initially have
            antibiotic resistance
        mut_prob (float in [0, 1]): mutation probability for the
            ResistantBacteria cells
        num_trials (int): number of simulation runs to execute

    Returns: a tuple of two lists of lists, or two 2D arrays
        populations (list of lists or 2D array): the total number of bacteria
            at each time step for each trial; total_population[i][j] is the
            total population for trial i at time step j
        resistant_pop (list of lists or 2D array): the total number of
            resistant bacteria at each time step for each trial;
            resistant_pop[i][j] is the number of resistant bacteria for
            trial i at time step j
    """
    timesteps_1 = 150
    timesteps_2 = 250
    bacteria =[]
    patient = TreatedPatient(bacteria, max_pop)
    populations =[]
    resistant_pop = []
    #num_trials = 1
    """ run the number of trials"""
    for p in range(num_trials):
        """ instantiate a list of SimpleBacteria """
        bacteria =[]
        rest_bac_pop = []
        for i in range(num_bacteria):
            bacteria.append(ResistantBacteria (birth_prob, death_prob,resistant,mut_prob))
        """instantiate a Patient using the list of SimpleBacteria"""
        patient = TreatedPatient(bacteria, max_pop)
        patient_time_steps = []
        """ simulate change for 150 timesteps then activated antibiotic and 300 time steps"""
        patient_time_steps.append(patient.get_total_pop())
        for i in range(timesteps_1):
            #print ("timesteps:", i)
            #print (patient.on_antibiotic)
            patient_time_steps.append(patient.update())
            rest_bac_pop.append(patient.get_resist_pop())
            #print ("patient list at time steps",i,"Number of trials", p, patient_time_steps)
        for i in range(timesteps_2):
            patient.set_on_antibiotic()
            #print (patient.on_antibiotic)
            patient_time_steps.append(patient.update())
            rest_bac_pop.append(patient.get_resist_pop())
        populations.insert(p,patient_time_steps)
        resistant_pop.insert(p,rest_bac_pop)
    #print("trial 0",populations[0],"\n", populations[0][1], "\ntrial 1", populations[1],"\n",populations[1][3])
    print ("95% conf int at step 299. Total population", calc_95_ci(populations, 299), "\n95% conf int at step 299. resistant bacteria", calc_95_ci(resistant_pop, 299))
    trial =0
    x_coords = []
    y_coords1 = []
    y_coords2 =[]
    for i in range(timesteps_1 + timesteps_2):
        x_coords.append(i)
        y_coords1.append(calc_pop_avg(populations, i))
        y_coords2.append(calc_pop_avg(resistant_pop,i))

    plt = make_two_curve_plot(x_coords,y_coords1, y_coords2,y_name1 ="Total population",y_name2 ="resistant", x_label="Timestep", y_label= "average population", title = "With an antibioic")
    return populations, resistant_pop

    pass  # TODO


# When you are ready to run the simulations, uncomment the next lines one
# at a time

total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100, max_pop=1000,birth_prob=0.3,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)

total_pop, resistant_pop = simulation_with_antibiotic(num_bacteria=100,
                                                      max_pop=1000,
                                                      birth_prob=0.17,
                                                      death_prob=0.2,
                                                      resistant=False,
                                                      mut_prob=0.8,
                                                      num_trials=50)
