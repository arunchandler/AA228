import numpy as np

class Piece():

    def __init__(self, x: int, y: int, team: str):
        self.isking = False
        self.team = team
        self.x = x
        self.y = y
    
class Player():

    def __init__(self, team: str, num_pieces: int):
        self.team = team
        self.num_pieces = num_pieces
        self.myturn = True

    def move():

        raise NotImplementedError