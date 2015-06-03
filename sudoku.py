import sys
#Puzzle:
#Holds sudoku boards and implemented sudoku strategies
class Puzzle:
    def __init__(self, inp = None, start = None):
        if start is not None:
            board = []
            for i in range(9):
                board.append([])
                for j in range(9):
                    board[-1].append([])
            self.board = board
            isCheck=[]
            for i in range(9):
                isCheck.append([])
                for j in range(9):
                    isCheck[-1].append(False)
            self.isCheck = isCheck
            exist=[]
            for i in range(10):
                exist.append([])
                for j in range(9):
                    exist[-1].append([])
                    for k in range(9):
                        exist[-1][-1].append(False)
            self.exist = exist
            row=[]
            col=[]
            for i in range(9):
                row.append([1,2,3,4,5,6,7,8,9])
                col.append([1,2,3,4,5,6,7,8,9])
            self.row = row
            self.col = col
            reg=[]
            for i in range(3):
                reg.append([])
                for j in range(3):
                    reg[-1].append([1,2,3,4,5,6,7,8,9])
            self.reg = reg
            for i in range(9):
                for j in range(9):
                    if len(start[i][j])==1:
                        board[i][j].append(start[i][j][0])
                        exist[start[i][j][0]][i][j]=True
                        isCheck[i][j]=True
                        row[i].remove(start[i][j][0])
                        col[j].remove(start[i][j][0])
                        reg[i/3][j/3].remove(start[i][j][0])
                    else:
                        for k in start[i][j]:
                            board[i][j].append(k)
                            exist[k][i][j]=True
        else:
            board = []
            for i in range(9):
                board.append([])
                for j in range(9):
                    board[-1].append([])
                    for k in range(1,10):
                        board[-1][-1].append(k)
                self.board = board
            isCheck=[]
            for i in board:
                isCheck.append([])
                for j in i:
                    isCheck[-1].append(False)
            self.isCheck = isCheck
            exist=[]
            for i in range(10):
                exist.append([])
                for j in range(9):
                    exist[-1].append([])
                    for k in range(9):
                        exist[-1][-1].append(True)
            self.exist = exist
            row=[]
            col=[]
            for i in range(9):
                row.append([1,2,3,4,5,6,7,8,9])
                col.append([1,2,3,4,5,6,7,8,9])
            self.row = row
            self.col = col
            reg=[]
            for i in range(3):
                reg.append([])
                for j in range(3):
                    reg[-1].append([1,2,3,4,5,6,7,8,9])
            self.reg = reg
        replace=[]
        if isinstance(inp,basestring):
            f=open(inp,'r')
            linecounter=0
            
            for j in range(9):
                line=f.readline()
                for i in range(9):
                    if line[i]!="o":
                        replace+=[(linecounter,i,int(line[i]))]
                linecounter+=1
        elif isinstance(inp,list):
            replace = inp
        for i,j,k in replace:
            board[i][j]=[k]
        self.initBoard = []
        for i in range(9):
            self.initBoard.append([])
            for j in range(9):
                self.initBoard[-1].append([])
                for k in board[i][j]:
                    self.initBoard[-1][-1].append(k)
        for i,j,k in replace:
            self.check(i,j)
        self.mode = "Easy"
        
    def check(self,i,j):
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        check = self.check
        xreg=(i/3)*3
        yreg=(j/3)*3
        x=i/3
        y=j/3
        if len(board[i][j])==1 and not isCheck[i][j]:
            checkQueue=[]
            isCheck[i][j]=True
            row[i].remove(board[i][j][0])
            col[j].remove(board[i][j][0])
            for k in range(1,10):
                if board[i][j][0]!=k:
                    exist[k][i][j]=False
            reg[x][y].remove(board[i][j][0])
            for k in range(9):
                if j!=k and board[i][j][0] in board[i][k]:
                    exist[board[i][j][0]][i][k]=False
                    board[i][k].remove(board[i][j][0])
                    if len(board[i][k])==1:
                        checkQueue+=[(i,k)]
            for k in range(9):
                if i!=k and board[i][j][0] in board[k][j]:
                    exist[board[i][j][0]][k][j]=False
                    board[k][j].remove(board[i][j][0])
                    if len(board[k][j])==1:
                        checkQueue+=[(k,j)]
            for o in range(3):
                for p in range(3):
                    m=xreg+o
                    n=yreg+p
                    if not (m==i and n==j) and board[i][j][0] in board[m][n]:
                        exist[board[i][j][0]][m][n]=False
                        board[m][n].remove(board[i][j][0])
                        if len(board[m][n])==1:
                            checkQueue+=[(m,n)]
            for o,p in checkQueue:
                check(o,p)
            return True
        return False
    def colCheck(self):
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        found=0
        for i in range(len(col)):
            for k in col[i]:
                counter=0
                store=(-1,-1)
                for l in range(9):
                    if exist[k][l][i]:
                        counter+=1
                        store=(l,i)
                        if counter>1:
                            break
                if counter==1:
                    board[store[0]][store[1]]=[k]
                    check(store[0],store[1])
                    found+=1
        return found
    def rowCheck(self):
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        found=0
        for i in range(len(row)):
            for k in row[i]:
                counter=0
                store=(-1,-1)
                for l in range(9):
                    if exist[k][i][l]:
                        counter+=1
                        store=(i,l)
                        if counter>1:
                            break
                if counter==1:
                    board[store[0]][store[1]]=[k]
                    check(store[0],store[1])
                    found+=1
        return found
    def regCheck(self):
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        found=0
        for i in range(len(reg)):
            for j in range(len(reg[0])):

                for k in reg[i][j]:
                    counter=0
                    store=(-1,-1)
                    for l in range(3):
                        for m in range(3):

                            if exist[k][i*3+l][j*3+m]:
                                counter+=1
                                store=(i*3+l,j*3+m)
                                if counter>1:
                                    break
                        if counter>1:
                            break
                    if counter==1:
                        board[store[0]][store[1]]=[k]
                        check(store[0],store[1])
                        found+=1
        return found
    def duoCheck(self):
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        update=0
        for k in range(1,10):
            rows=[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
            cols=[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
            isUpdated=False
            for i in range(3):
                for j in range(3):
                    if k not in reg[i][j]:
                        rows[i][j]=-2
                        cols[i][j]=-2
                    else:
                        row1=[]
                        col1=[]
                        for l in range(3):
                            if exist[k][i*3+l][j*3] or exist[k][i*3+l][j*3+1] or exist[k][i*3+l][j*3+2]:
                                row1.append(l)
                            if exist[k][i*3][j*3+l] or exist[k][i*3+1][j*3+l] or exist[k][i*3+2][j*3+l]:
                                col1.append(l)
                        if len(row1)==2:
                            if row1[1]==2:
                                if row1[0]==0:
                                    rows[i][j]=1
                                else:
                                    rows[i][j]=0
                            else:
                                rows[i][j]=2
                        else:
                            rows[i][j]=-2
                        if len(col1)==2:
                            if col1[1]==2:
                                if col1[0]==0:
                                    cols[i][j]=1
                                else:
                                    cols[i][j]=0
                            else:
                                cols[i][j]=2
                        else:
                            cols[i][j]=-2
            for p in [rows,cols]:
                for i in range(3):
                    for a,b,c in [(0,1,2),(0,2,1),(1,2,0)]:
                        if (p[i][a]==p[i][b]>=0 and p is rows) or (p[a][i]==p[b][i]>=0 and p is cols):
                            for m in range(3):
                                if p is rows and m==p[i][a]:
                                    continue
                                for n in range(3):
                                    if p is cols and n==p[a][i]:
                                        continue
                                    x=0
                                    y=0
                                    if p is rows:
                                        x=i*3+m
                                        y=c*3+n
                                    else:
                                        x=c*3+m
                                        y=i*3+n
                                    if exist[k][x][y]:

                                        isUpdated=True
                                        exist[k][x][y]=False
                                        board[x][y].remove(k)
                                        if len(board[x][y])==1:
                                            check(x,y)
            if isUpdated:
                update+=1
        return update                       
    def canCheck(self):
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        update=0
        for i in range(len(reg)):
            for j in range(len(reg[0])):
                for k in reg[i][j]:
                    row1=-1
                    col1=-1
                    stop=False
                    for l in range(3):
                        for m in range(3):
                            if exist[k][i*3+l][j*3+m]:
                                if row1==-1:
                                    row1=i*3+l
                                elif row1!=-2 and row1!=i*3+l:
                                    row1=-2
                                    if col1==-2:
                                        stop=True
                                        break
                                if col1==-1:
                                    col1=j*3+m
                                elif col1!=-2 and col1!=j*3+m:
                                    col1=-2
                                    if row1==-2:
                                        stop=True
                                        break
                        if stop:
                            break
                    isUpdated=False
                    if row1>=0:
                        for l in range(9):
                            if j*3<=l<=j*3+2:
                                continue
                            if exist[k][row1][l]:
                                isUpdated=True
                                exist[k][row1][l]=False
                                board[row1][l].remove(k)
                                if len(board[row1][l])==1:
                                    check(row1,l)
                    if col1>=0:
                        for l in range(9):
                            if i*3<=l<=i*3+2:
                                continue
                            if exist[k][l][col1]:
                                isUpdated=True
                                exist[k][l][col1]=False
                                board[l][col1].remove(k)
                                if len(board[l][col1])==1:
                                    check(l,col1)
                    if isUpdated:
                        update+=1
        return update
    def nakCheck(self):
        comb = self.comb
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        update=0
        for i in range(9):
            combos=comb(row[i],2)+comb(row[i],3)+comb(row[i],4)
            for k in combos:
                isUpdated=False
                counter=[]
                for j in range(9):
                    isNaked=True
                    for l in board[i][j]:
                        if l not in k:
                            isNaked=False
                            break
                    if isNaked:
                        counter+=[j]
                if len(counter)==len(k):
                    for j in range(9):
                        if j in counter:
                            continue
                        else:
                            x=i
                            y=j
                            for l in k:
                                if exist[l][x][y]:
                                    isUpdated=True
                                    exist[l][x][y]=False
                                    board[x][y].remove(l)
                                    if len(board[x][y])==1:
                                        check(x,y)
                if isUpdated:
                    update+=1
        for j in range(9):
            combos=comb(col[j],2)+comb(col[j],3)+comb(col[j],4)
            for k in combos:
                isUpdated=False
                counter=[]
                for i in range(9):
                    isNaked=True
                    for l in board[i][j]:
                        if l not in k:
                            isNaked=False
                            break
                    if isNaked:
                        counter+=[i]
                if len(counter)==len(k):
                    for i in range(9):
                        if i in counter:
                            continue
                        else:
                            x=i
                            y=j
                            for l in k:
                                if exist[l][x][y]:
                                    isUpdated=True
                                    exist[l][x][y]=False
                                    board[x][y].remove(l)
                                    if len(board[x][y])==1:
                                        check(x,y)
        for a,b in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:
            combos=comb(reg[a][b],2)+comb(reg[a][b],3)+comb(reg[a][b],4)
            for k in combos:
                isUpdated=False
                counter=[]
                for i,j in [(3*a+x,3*b+y) for x,y in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]]:
                    isNaked=True
                    for l in board[i][j]:
                        if l not in k:
                            isNaked=False
                            break
                    if isNaked:
                        counter+=[(i,j)]
                if len(counter)==len(k):
                    for i,j in [(3*a+x,3*b+y) for x,y in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]]:
                        if (i,j) in counter:
                            continue
                        else:
                            x=i
                            y=j
                            for l in k:
                                if exist[l][x][y]:
                                    isUpdated=True
                                    exist[l][x][y]=False
                                    board[x][y].remove(l)
                                    if len(board[x][y])==1:
                                        check(x,y)
                if isUpdated:
                    update+=1
        return update
    def hidCheck(self):
        comb = self.comb
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist

        update=0
        for i in range(9):
            combos=comb(row[i],2)+comb(row[i],3)+comb(row[i],4)
            for k in combos:
                isUpdated=False
                counter=[]
                for j in range(9):
                    isHidden=False
                    for l in k:
                        if l in board[i][j]:
                            isHidden=True
                            break
                    if isHidden:
                        counter+=[j]
                if len(counter)==len(k):
                    for j in counter:
                        x=i
                        y=j
                        for l in board[x][y]:
                            if l not in k:
                                isUpdated=True
                                exist[l][x][y]=False
                                board[x][y].remove(l)
                                if len(board[x][y])==1:
                                    check(x,y)
                if isUpdated:
                    update+=1
        for j in range(9):
            combos=comb(col[j],2)+comb(col[j],3)+comb(col[j],4)
            for k in combos:
                isUpdated=False
                counter=[]
                for i in range(9):
                    isHidden=False
                    for l in k:
                        if l in board[i][j]:
                            isHidden=True
                            break
                    if isHidden:
                        counter+=[i]
                if len(counter)==len(k):
                    for i in counter:
                        x=i
                        y=j
                        for l in board[x][y]:
                            if l not in k:
                                isUpdated=True
                                exist[l][x][y]=False
                                board[x][y].remove(l)
                                if len(board[x][y])==1:
                                    check(x,y)
        for a,b in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]:
            combos=comb(reg[a][b],2)+comb(reg[a][b],3)+comb(reg[a][b],4)
            for k in combos:
                isUpdated=False
                counter=[]
                for i,j in [(3*a+x,3*b+y) for x,y in [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]]:
                    isHidden=False
                    for l in k:
                        if l in board[i][j]:
                            isHidden=True
                            break
                    if isHidden:
                        counter+=[(i,j)]
                if len(counter)==len(k):
                    for i,j in counter:
                        x=i
                        y=j
                        for l in board[x][y]:
                            if l not in k:
                                isUpdated=True
                                exist[l][x][y]=False
                                board[x][y].remove(l)
                                if len(board[x][y])==1:
                                    check(x,y)
                if isUpdated:
                    update+=1
        return update
    def xwiCheck(self):
        check = self.check
        board = self.board
        isCheck = self.isCheck
        row = self.row
        col = self.col
        reg = self.reg 
        exist = self.exist
        update=0
        for i1 in range(9):
            for i2 in range(9):
                if i1!=i2:
                    for k in range(9):
                        if k in row[i1] and k in row[i2]:
                            isUpdated=False
                            isXwing=True
                            cand = []
                            for j in range(9):
                                if exist[k][i1][j] and exist[k][i2][j]:
                                    cand+=[j]
                                    if len(cand)>2:
                                        isXwing=False
                                        break
                                if (exist[k][i1][j] and not exist[k][i2][j]) or (not exist[k][i1][j] and exist[k][i2][j]):
                                    isXwing=False
                                    break
                            if isXwing and len(cand)==2:
                                for j in cand:
                                    for i in range(9):
                                        if i==i1 or i==i2:
                                            continue
                                        x=i
                                        y=j
                                        if exist[k][x][y]:
                                            isUpdated=True
                                            exist[k][x][y]=False
                                            board[x][y].remove(k)
                                            if len(board[x][y])==1:
                                                check(x,y)
                            if isUpdated:
                                update+=1
        for j1 in range(9):
            for j2 in range(9):
                if j1!=j2:
                    for k in range(9):
                        if k in col[j1] and k in col[j2]:
                            isUpdated=False
                            isXwing=True
                            cand = []
                            for i in range(9):
                                if exist[k][i][j1] and exist[k][i][j2]:
                                    cand+=[i]
                                    if len(cand)>2:
                                        isXwing=False
                                        break
                                if (exist[k][i][j1] and not exist[k][i][j2]) or (not exist[k][i][j1] and exist[k][i][j2]):
                                    isXwing=False
                                    break
                            if isXwing and len(cand)==2:

                                for i in cand:
                                    for j in range(9):
                                        if j==j1 or j==j2:
                                            continue
                                        x=i
                                        y=j
                                        if exist[k][x][y]:
                                            isUpdated=True
                                            exist[k][x][y]=False
                                            board[x][y].remove(k)
                                            if len(board[x][y])==1:
                                                check(x,y)
                            if isUpdated:
                                update+=1
        return update
    def comb(self,arr,desired):
        length=len(arr)
        if length<=desired:
            return [arr]
        else:
            combined=[]
            for i in range(desired+1):
                combined+=[arr[:i]+x for x in self.comb(arr[i+1:],desired-i)]
            return combined
    def printExist(self):
        print "-------------------------------------------------------------------------------------------------------------"
        for j in range(9):
            sys.stdout.write("|")
            for i in range(1,10):
                for k in range(9):
                    if self.exist[i][j][k]:
                        sys.stdout.write(str(i))
                    else:
                        sys.stdout.write(" ")
                    if k%3==2:
                        sys.stdout.write("|")
            print""
            if j%3==2:
                print "-------------------------------------------------------------------------------------------------------------"
    def printBoard(self):
        print "      0         1         2          3         4         5          6         7         8"
        print " ----------------------------------------------------------------------------------------------"
        for i in range(9):
            sys.stdout.write(str(i)+"|")
            for j in range(9):
                for k in range(1,10):
                    if k in self.board[i][j]:
                        sys.stdout.write(str(k))
                    else:
                        sys.stdout.write(" ")
                if j%3==2:
                    sys.stdout.write("||")
                else:
                    sys.stdout.write("|")
            print ""
            if i%3==2:
                print " ----------------------------------------------------------------------------------------------"
    def printSmallBoard(self):
        print self.mode
        store=""
        for i in range(9):
            for j in range(9):
                if len(self.board[i][j])>1:
                    store+="o"
                else:
                    store+=str(self.board[i][j][0])
            store+="\n"
        print store
        return store
    def placeCheck(self):
        check = self.check
        found=0
        for i in range(9):
            for j in range(9):
                if check(i,j):
                    found+=1
        return found
    def checkWin(self):
        for i in range(9):
            for j in range(9):
                if not self.isCheck[i][j]:
                    return False
        return True
    def checkInvalid(self):
        for i in self.board:
            for j in i:
                if len(j)==0:
                    return True
        return False
    def guess(self):
        self.mode = "Insane"
        for i in range(9):
            for j in range(9):
                if len(self.board[i][j])==1:
                    continue
                elif len(self.board[i][j])==0:
                    return False
                else:
                    for k in self.board[i][j]:
                        newpuzzle = Puzzle(inp = [(i,j,k)],start=self.board)
                        if newpuzzle.solveBoard():
                            self.board = newpuzzle.board
                            self.isCheck = newpuzzle.isCheck
                            self.exist = newpuzzle.exist
                            self.row = newpuzzle.row
                            self.col = newpuzzle.col
                            self.reg = newpuzzle.reg

                            return True
                        elif newpuzzle.checkInvalid():
                            continue
                        else:
                            if newpuzzle.guess():
                                self.board = newpuzzle.board
                                self.isCheck = newpuzzle.isCheck
                                self.exist = newpuzzle.exist
                                self.row = newpuzzle.row
                                self.col = newpuzzle.col
                                self.reg = newpuzzle.reg
                                return True
                            else:
                                return False
        return False
    def solveBoardWithGuessing(self):
        self.solveBoard()
        if self.checkWin():
            return True
        return self.guess()
        
    def printInitBoard(self):
        print self.mode
        store=""
        for i in range(9):
            for j in range(9):
                if len(self.initBoard[i][j])>1:
                    store+="o"
                else:
                    store+=str(self.initBoard[i][j][0])
            store+="\n"
        print store
        return store
    def solveBoard(self):
        regCheck = self.regCheck
        rowCheck = self.rowCheck
        colCheck = self.colCheck
        placeCheck = self.placeCheck
        canCheck = self.canCheck
        duoCheck = self.duoCheck
        nakCheck = self.nakCheck
        hidCheck = self.hidCheck
        printSmallBoard = self.printSmallBoard
        xwiCheck = self.xwiCheck
        checkWin = self.checkWin
        while True:
            regcount= regCheck()
            if regcount>0:
                pass
            rowcount= rowCheck()
            if rowcount>0:
                pass
            colcount= colCheck()
            if colcount>0:
                pass
            placecount= placeCheck()
            if placecount>0:
                pass
            if self.mode=="Medium" or self.mode=="Hard":
                cancount=canCheck()
                if cancount>0:
                    pass
                duocount= duoCheck()
                if duocount>0:
                    pass
            if self.mode=="Hard":
                nakcount=nakCheck()
                if nakcount>0:
                    pass
                hidcount=hidCheck()
                if hidcount>0:
                    pass
            if self.mode=="Very Hard":
                xwicount=xwiCheck()
                if xwicount>0:
                    pass
            if checkWin():
                return True
            if self.mode=="Easy" and regcount+rowcount+colcount+placecount==0:
                self.mode="Medium"
            elif self.mode=="Medium" and regcount+rowcount+colcount+placecount+cancount+duocount==0:
                self.mode="Hard"
            elif self.mode=="Hard" and regcount+placecount+rowcount+colcount+cancount+duocount+nakcount+hidcount==0:
                self.mode="Very Hard"
            elif self.mode=="Very Hard" and regcount+placecount+rowcount+colcount+cancount+duocount+nakcount+hidcount+xwicount==0:
                break
        return False

def run():
    puzzle = Puzzle(inp='sudokuinput.txt')
    newpuzzle = Puzzle(start=puzzle.board)
    newpuzzle.solveBoardWithGuessing()
    newpuzzle.printSmallBoard()
    
