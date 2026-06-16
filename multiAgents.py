# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostStates =  [ghostState for ghostState in successorGameState.getGhostStates() if ghostState.scaredTimer == 0]
        scaredGhostStates = [ghostState for ghostState in successorGameState.getGhostStates() if ghostState.scaredTimer > 0]

        foodDistance = min([manhattanDistance(newPos, food) for food in newFood.asList()]) if newFood.asList() else 0
        ghostDistance = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in ghostStates]) if ghostStates else float('inf')
        scaredGhostDistance = min([manhattanDistance(newPos, ghostState.getPosition()) for ghostState in scaredGhostStates]) if scaredGhostStates else float('inf')
        foodCount = successorGameState.getNumFood()
        powerPelletDistance = min([manhattanDistance(newPos, pellet) for pellet in successorGameState.getCapsules()]) if successorGameState.getCapsules() else float('inf')

        nearGhostCount = sum(1 for ghostState in ghostStates if manhattanDistance(newPos, ghostState.getPosition()) < 4)

        score = successorGameState.getScore()

        SCORE_WEIGHT = 9.0
        FOOD_WEIGHT = 1.3 * SCORE_WEIGHT
        GHOST_WEIGHT = -6.0 * SCORE_WEIGHT
        SCARED_GHOST_WEIGHT = 4.5 * SCORE_WEIGHT
        FOOD_COUNT_WEIGHT = -1.0 * SCORE_WEIGHT
        POWER_PELLET_WEIGHT = -2.6 * SCORE_WEIGHT

        if foodDistance > 0:
            score += FOOD_WEIGHT / foodDistance
        else:
            score += 500.0

        if ghostDistance > 0:
            score += GHOST_WEIGHT / ghostDistance

        if scaredGhostDistance > 0:
            score += SCARED_GHOST_WEIGHT / scaredGhostDistance

        score += FOOD_COUNT_WEIGHT * foodCount

        if nearGhostCount >= 2 and powerPelletDistance < 10:
            score += POWER_PELLET_WEIGHT / powerPelletDistance

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        bestAction = None
        bestValue = float('-inf')

        for action in gameState.getLegalActions(0):
            successorGameState = gameState.generateSuccessor(0, action)
            value = self.actionTree(successorGameState, 1, self.depth)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction

    def actionTree(self, gameState, agentIndex, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            bestValue = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = self.actionTree(successorGameState, (agentIndex + 1) % gameState.getNumAgents(), depth)
                bestValue = max(bestValue, value)
            return bestValue
        else:
            if (agentIndex + 1) == gameState.getNumAgents():
                depth -= 1

            bestValue = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = self.actionTree(successorGameState, (agentIndex + 1) % gameState.getNumAgents(), depth)
                bestValue = min(bestValue, value)
            return bestValue
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        
        bestAction = None
        bestValue = float('-inf')

        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            successorGameState = gameState.generateSuccessor(0, action)
            value = self.actionTree(successorGameState, 1, self.depth, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction

    def actionTree(self, gameState, agentIndex, depth, alpha, beta):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            bestValue = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = self.actionTree(successorGameState, (agentIndex + 1) % gameState.getNumAgents(), depth, alpha, beta)
                bestValue = max(bestValue, value)
                if bestValue > beta:
                    return bestValue
                alpha = max(alpha, bestValue)
            return bestValue
        else:
            if (agentIndex + 1) == gameState.getNumAgents():
                depth -= 1

            bestValue = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = self.actionTree(successorGameState, (agentIndex + 1) % gameState.getNumAgents(), depth, alpha, beta)
                bestValue = min(bestValue, value)
                if bestValue < alpha:
                    return bestValue
                beta = min(beta, bestValue)
            return bestValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
