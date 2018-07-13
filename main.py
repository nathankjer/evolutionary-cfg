import math

from optimizers import EvolutionaryCFG

def evaluate(expression):
    return math.fabs(eval(expression))

def main():
    symbols = ['x','y','z','T','I','S']
    grammar = {
        'x':[str(float(x)) for x in xrange(1,10)],
        'y':[str(float(x)) for x in xrange(1,10)],
        'z':[str(float(x)) for x in xrange(1,10)],
        'T':['x','y','z'],
        'I':['I+I','I-I','I*I','I/I','(I)'] + ['T']*5,
        'S':['I+I','I-I','I*I','I/I']
    }

    optimizer = EvolutionaryCFG(symbols, grammar, evaluate, population_size=300, num_attributes=7)
    optimizer.evolve(generations=4)

if __name__ == '__main__':
    main()
