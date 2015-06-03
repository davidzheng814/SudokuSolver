

import random
import sudoku
def generateFullBoard(index=0,board=[]):
	if board==[]:
		for i in range(9):
			board.append([])
			for j in range(9):
				board[-1].append(range(1,10))
	i=index/9
	j=index%9
	a=i/3
	b = j/3
	if index==81:
		return True
	if len(board[i][j])==0:
			return False
	shuffle=range(1,10)
	random.shuffle(shuffle)
	for rand in shuffle:
		if rand not in board[i][j]:
			continue
		store=[]
		for k in board[i][j]:
			if k!=rand:
				store+=[(i,j,k)]
		board[i][j] = [rand]
		for k in range(i+1,9):
			if rand in board[k][j]:
				board[k][j].remove(rand)
				store+=[(k,j,rand)]
		for k in range(j+1,9):
			if rand in board[i][k]:
				board[i][k].remove(rand)
				store+=[(i,k,rand)]
		for k,l in [(a*3+x,b*3+y) for x,y in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]]:
			if not (k==i and l==j):
				if rand in board[k][l]:
					board[k][l].remove(rand)
					store+=[(k,l,rand)]
		if generateFullBoard(index=index+1,board=board):
			return board
		else:
			for k,l,m in store:
				board[k][l].append(m)
			continue
	return False
def convertToReplace(board):
	replace = []
	for i in range(9):
		for j in range(9):
			if len(board[i][j])==1:
				replace+=[(i,j,board[i][j][0])]
	return replace
def generateNewPuzzle():
	board = generateFullBoard()
	replace = convertToReplace(board)
	while True:
		index = random.randint(0,len(replace)-1)
		removed = replace.pop(index)

		puzzle = sudoku.Puzzle(replace)
		if puzzle.solveBoard():
			continue
		else:
			replace.append(removed)
			indices = range(len(replace))
			random.shuffle(indices)
			success=True
			for i in indices:
				removed = replace.pop(i)
				puzzle = sudoku.Puzzle(replace)
				if puzzle.solveBoard():
					success=False
					break
				replace.insert(i,removed)
			if not success:
				continue
			puzzle = sudoku.Puzzle(replace)
			puzzle.solveBoard()
			return puzzle.printInitBoard()
generateNewPuzzle()
