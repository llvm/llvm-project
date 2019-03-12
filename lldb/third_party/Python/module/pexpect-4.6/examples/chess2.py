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
import sys
import time

class Chess:

        def __init__(self, engine = "/usr/local/bin/gnuchess -a -h 1"):
                self.child = pexpect.spawn (engine)
                self.term = ANSI.ANSI ()

                #self.child.expect ('Chess')
                #if self.child.after != 'Chess':
                #        raise IOError, 'incompatible chess program'
                #self.term.process_list (self.child.before)
                #self.term.process_list (self.child.after)

                self.last_computer_move = ''

        def read_until_cursor (self, r,c, e=0):
            '''Eventually something like this should move into the screen class or
            a subclass. Maybe a combination of pexpect and screen...
            '''
            fout = open ('log','a')
            while self.term.cur_r != r or self.term.cur_c != c:
                try:
                    k = self.child.read(1, 10)
                except Exception as e:
                    print('EXCEPTION, (r,c):(%d,%d)\n' %(self.term.cur_r, self.term.cur_c))
                    sys.stdout.flush()
                self.term.process (k)
                fout.write ('(r,c):(%d,%d)\n' %(self.term.cur_r, self.term.cur_c))
                fout.flush()
                if e:
                    sys.stdout.write (k)
                    sys.stdout.flush()
                if self.term.cur_r == r and self.term.cur_c == c:
                    fout.close()
                    return 1
            print('DIDNT EVEN HIT.')
            fout.close()
            return 1

        def expect_region (self):
            '''This is another method that would be moved into the
            screen class.
            '''
            pass
        def do_scan (self):
            fout = open ('log','a')
            while 1:
                c = self.child.read(1,10)
                self.term.process (c)
                fout.write ('(r,c):(%d,%d)\n' %(self.term.cur_r, self.term.cur_c))
                fout.flush()
                sys.stdout.write (c)
                sys.stdout.flush()

        def do_move (self, move, e = 0):
                time.sleep(1)
                self.read_until_cursor (19,60, e)
                self.child.sendline (move)

        def wait (self, color):
            while 1:
                r = self.term.get_region (14,50,14,60)[0]
                r = r.strip()
                if r == color:
                    return
                time.sleep (1)

        def parse_computer_move (self, s):
                i = s.find ('is: ')
                cm = s[i+3:i+9]
                return cm
        def get_computer_move (self, e = 0):
                time.sleep(1)
                self.read_until_cursor (19,60, e)
                time.sleep(1)
                r = self.term.get_region (17,50,17,62)[0]
                cm = self.parse_computer_move (r)
                return cm

        def switch (self):
                print('switching')
                self.child.sendline ('switch')

        def set_depth (self, depth):
                self.child.sendline ('depth')
                self.child.expect ('depth=')
                self.child.sendline ('%d' % depth)

        def quit(self):
                self.child.sendline ('quit')

def LOG (s):
    print(s)
    sys.stdout.flush ()
    fout = open ('moves.log', 'a')
    fout.write (s + '\n')
    fout.close()

print('Starting...')

black = Chess()
white = Chess()
white.read_until_cursor (19,60,1)
white.switch()

done = 0
while not done:
    white.wait ('Black')
    move_white = white.get_computer_move(1)
    LOG ( 'move white:'+ move_white )

    black.do_move (move_white)
    black.wait ('White')
    move_black = black.get_computer_move()
    LOG ( 'move black:'+ move_black )

    white.do_move (move_black, 1)

g.quit()


