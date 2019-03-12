#!/usr/bin/env python

'''This demonstrates controlling a screen oriented application (curses).
It starts two instances of gnuchess and then pits them against each other.

PEXPECT LICENSE

    This license is approved by the OSI and FSF as GPL-compatible.
        http://opensource.org/licenses/isc-license.txt

    Copyright (c) 2012, Noah Spurrier <noah@noah.org>
    PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE THIS SOFTWARE FOR ANY
    PURPOSE WITH OR WITHOUT FEE IS HEREBY GRANTED, PROVIDED THAT THE ABOVE
    COPYRIGHT NOTICE AND THIS PERMISSION NOTICE APPEAR IN ALL COPIES.
    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

'''

from __future__ import print_function

from __future__ import absolute_import

import pexpect
import ANSI

REGEX_MOVE = r'(?:[a-z]|\x1b\[C)(?:[0-9]|\x1b\[C)(?:[a-z]|\x1b\[C)(?:[0-9]|\x1b\[C)'
REGEX_MOVE_PART = r'(?:[0-9]|\x1b\[C)(?:[a-z]|\x1b\[C)(?:[0-9]|\x1b\[C)'

class Chess:

    def __init__(self, engine = "/usr/local/bin/gnuchess -a -h 1"):
        self.child = pexpect.spawn (engine)
        self.term = ANSI.ANSI ()

        self.child.expect ('Chess')
        if self.child.after != 'Chess':
            raise IOError('incompatible chess program')
        self.term.process_list (self.before)
        self.term.process_list (self.after)
        self.last_computer_move = ''

    def read_until_cursor (self, r,c):
        while 1:
            self.child.read(1, 60)
            self.term.process (c)
            if self.term.cur_r == r and self.term.cur_c == c:
                return 1

    def do_first_move (self, move):
        self.child.expect ('Your move is')
        self.child.sendline (move)
        self.term.process_list (self.before)
        self.term.process_list (self.after)
        return move

    def do_move (self, move):
        self.read_until_cursor (19,60)
        self.child.sendline (move)
        return move

    def get_first_computer_move (self):
        self.child.expect ('My move is')
        self.child.expect (REGEX_MOVE)
        return self.child.after

    def get_computer_move (self):
        print('Here')
        i = self.child.expect ([r'\[17;59H', r'\[17;58H'])
        print(i)
        if i == 0:
            self.child.expect (REGEX_MOVE)
            if len(self.child.after) < 4:
                self.child.after = self.child.after + self.last_computer_move[3]
        if i == 1:
            self.child.expect (REGEX_MOVE_PART)
            self.child.after = self.last_computer_move[0] + self.child.after
        print('', self.child.after)
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
import sys
print('Starting...')
white = Chess()
white.child.echo = 1
white.child.expect ('Your move is')
white.set_depth(2)
white.switch()

move_white = white.get_first_computer_move()
print('first move white:', move_white)

white.do_move ('e7e5')
move_white = white.get_computer_move()
print('move white:', move_white)
white.do_move ('f8c5')
move_white = white.get_computer_move()
print('move white:', move_white)
white.do_move ('b8a6')
move_white = white.get_computer_move()
print('move white:', move_white)

sys.exit(1)



black = Chess()
white = Chess()
white.child.expect ('Your move is')
white.switch()

move_white = white.get_first_computer_move()
print('first move white:', move_white)

black.do_first_move (move_white)
move_black = black.get_first_computer_move()
print('first move black:', move_black)

white.do_move (move_black)

done = 0
while not done:
    move_white = white.get_computer_move()
    print('move white:', move_white)

    black.do_move (move_white)
    move_black = black.get_computer_move()
    print('move black:', move_black)

    white.do_move (move_black)
    print('tail of loop')

g.quit()
