import random, math
random.seed(64)

from deap import base
from deap import creator
from deap import tools

class EvolutionaryCFG:
    '''
    Evolutionary Context-Free Grammar
    '''

    def __init__(self, symbols, grammar, evaluation_function, population_size=20, num_attributes=30):

        self.symbols = symbols
        self.grammar = grammar
        attribute_size = max([len(x) for x in grammar.values()])
        self.evaluation_function = evaluation_function
        
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register('attr_int', random.randint, 0, attribute_size-1)
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.attr_int, num_attributes)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('evaluate', self.evaluate)
        self.toolbox.register('mate', tools.cxOnePoint)
        self.toolbox.register('mutate', tools.mutUniformInt, low=0, up=255, indpb=0.05)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

        self.population = self.toolbox.population(n=population_size)
        self.mate_prob = 0.5
        self.mutant_prob = 0.2
        
        # Prevents re-evaluation of the same individuals (assumes determinism)
        self.score_cache = {}

    def express(self, individual):
        expression = 'S'
        for branch in individual:
            expression_old = expression
            for symbol in self.symbols:
                if symbol in expression:
                    i = expression.rindex(symbol)
                    rules = self.grammar[symbol]
                    replacement = rules[branch % len(rules)]
                    expression = expression[:i] + replacement + expression[i+1:]
                    break
            if expression == expression_old:
                break
        return expression
        
    def evaluate(self, individual):
        expression = self.express(individual)
        if expression not in self.score_cache.keys():
            if any([symbol in expression for symbol in self.symbols]):
                score = float('-inf')
            else:
                try:
                    score = self.evaluation_function(expression)
                except:
                    score = float('-inf')
            self.score_cache[expression] = score
        else:
            score = self.score_cache[expression]
        return score,
    
    def evolve(self, generations):
        print('Start of evolution')
        
        # Evaluate the entire population
        print('\tGeneration 0')
        fitnesses = list(map(self.toolbox.evaluate, self.population))
        for ind, fit in zip(self.population, fitnesses):
            ind.fitness.values = fit
        
        print('\t\tEvaluated %i individuals' % len(self.population))
    
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in self.population if ind.fitness.values[0] != float('-inf')]
        length = len(self.population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print('\t\t\tMin %s' % min(fits))
        print('\t\t\tMax %s' % max(fits))
        print('\t\t\tAvg %s' % mean)
        print('\t\t\tStd %s' % std)
        
        for generation in xrange(1,generations+1):
            print('\tGeneration %i' % generation)
            
            # Select the next generation individuals
            offspring = self.toolbox.select(self.population, len(self.population))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
    
                # cross two individuals with probability self.mate_prob
                if random.random() < self.mate_prob:
                    self.toolbox.mate(child1, child2)
    
                    # fitness values of the children must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            # mutate an individual with probability self.mutant_prob
            for mutant in offspring:
                if random.random() < self.mutant_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            print('\t\tEvaluated %i individuals' % len(invalid_ind))
            
            # The population is entirely replaced by the offspring
            self.population[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.population if ind.fitness.values[0] != float('-inf')]
            length = len(self.population)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            print('\t\t\tMin %s' % min(fits))
            print('\t\t\tMax %s' % max(fits))
            print('\t\t\tAvg %s' % mean)
            print('\t\t\tStd %s' % std)
        
        print('Done!')
        
        best_ind = tools.selBest(self.population, 1)[0]
        print('Best individual is\n\t%s\n\t%s\n\nResult:%s' % (best_ind, self.express(best_ind), best_ind.fitness.values))
