#!/usr/bin/env python

'''This demonstrates controlling a screen oriented application (curses).
It starts two instances of gnuchess and then pits them against each other.
'''

import pexpect
import string
import ANSI

REGEX_MOVE = '(?:[a-z]|\x1b\[C)(?:[0-9]|\x1b\[C)(?:[a-z]|\x1b\[C)(?:[0-9]|\x1b\[C)'
REGEX_MOVE_PART = '(?:[0-9]|\x1b\[C)(?:[a-z]|\x1b\[C)(?:[0-9]|\x1b\[C)'

class Chess:

	def __init__(self, engine = "/usr/local/bin/gnuchess -a -h 1"):
		self.child = pexpect.spawn (engine)
                self.term = ANSI.ANSI ()
             
#		self.child.expect ('Chess')
	#	if self.child.after != 'Chess':
	#		raise IOError, 'incompatible chess program'
        #        self.term.process_list (self.before)
        #        self.term.process_list (self.after)
		self.last_computer_move = ''
        def read_until_cursor (self, r,c):
            fout = open ('log','a')
            while 1:
                k = self.child.read(1, 10)
                self.term.process (k)
                fout.write ('(r,c):(%d,%d)\n' %(self.term.cur_r, self.term.cur_c))
                fout.flush()
                if self.term.cur_r == r and self.term.cur_c == c:
                    fout.close()
                    return 1
                sys.stdout.write (k)
                sys.stdout.flush()

	def do_scan (self):
            fout = open ('log','a')
            while 1:
                c = self.child.read(1,10)
                self.term.process (c)
                fout.write ('(r,c):(%d,%d)\n' %(self.term.cur_r, self.term.cur_c))
                fout.flush()
                sys.stdout.write (c)
                sys.stdout.flush()

	def do_move (self, move):
                self.read_until_cursor (19,60)
		self.child.sendline (move)
		return move
	
	def get_computer_move (self):
		print 'Here'
		i = self.child.expect (['\[17;59H', '\[17;58H'])
		print i
		if i == 0:
			self.child.expect (REGEX_MOVE)
			if len(self.child.after) < 4:
				self.child.after = self.child.after + self.last_computer_move[3]
		if i == 1:
			self.child.expect (REGEX_MOVE_PART)
			self.child.after = self.last_computer_move[0] + self.child.after
		print '', self.child.after
		self.last_computer_move = self.child.after
		return self.child.after

	def switch (self):
		self.child.sendline ('switch')

	def set_depth (self, depth):
		self.child.sendline ('depth')
		self.child.expect ('depth=')
		self.child.sendline ('%d' % depth)

	def quit(self):
		self.child.sendline ('quit')
import sys, os
print 'Starting...'
white = Chess()
white.do_move('b2b4')
white.read_until_cursor (19,60)
c1 = white.term.get_abs(17,58)
c2 = white.term.get_abs(17,59)
c3 = white.term.get_abs(17,60)
c4 = white.term.get_abs(17,61)
fout = open ('log','a')
fout.write ('Computer:%s%s%s%s\n' %(c1,c2,c3,c4))
fout.close()
white.do_move('c2c4')
white.read_until_cursor (19,60)
c1 = white.term.get_abs(17,58)
c2 = white.term.get_abs(17,59)
c3 = white.term.get_abs(17,60)
c4 = white.term.get_abs(17,61)
fout = open ('log','a')
fout.write ('Computer:%s%s%s%s\n' %(c1,c2,c3,c4))
fout.close()
white.do_scan ()

#white.do_move ('b8a6')
#move_white = white.get_computer_move()
#print 'move white:', move_white

sys.exit(1)



black = Chess()
white = Chess()
white.child.expect ('Your move is')
white.switch()

move_white = white.get_first_computer_move()
print 'first move white:', move_white

black.do_first_move (move_white)
move_black = black.get_first_computer_move()
print 'first move black:', move_black

white.do_move (move_black)

done = 0
while not done:
    move_white = white.get_computer_move()
    print 'move white:', move_white

    black.do_move (move_white)
    move_black = black.get_computer_move()
    print 'move black:', move_black
   
    white.do_move (move_black)
    print 'tail of loop'

g.quit()


