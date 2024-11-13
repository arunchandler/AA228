import numpy as np
import functions as func
import objects as obj

class CheckersGame():

    def __init__(self, Player1: obj.Player, Player2: obj.Player):
        self.board = self.initialize_board()
        self.current_player = Player1

        raise NotImplementedError
    
    def initialize_board():

        raise NotImplementedError
    
    def get_reward():
        #dependent on row piece is on, if pieces can be taken

        raise NotImplementedError
    
    def update_board():

        raise NotImplementedError

    def get_legal_moves():

        raise NotImplementedError
    
    def is_legal():

        raise NotImplementedError
    
    def was_captured():

        raise NotImplementedError
    
    def print_board():

        raise NotImplementedError
    
    def is_game_over():
        raise NotImplementedError
    
def play():

    raise NotImplementedError
    
if __name__ == "__main__":
    play()