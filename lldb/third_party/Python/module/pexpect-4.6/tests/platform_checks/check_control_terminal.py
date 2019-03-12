#!/usr/bin/env python

import termios, fcntl, struct, os, sys

def getwinsize():	
	s = struct.pack("HHHH", 0, 0, 0, 0)
	x = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, s)
	rows, cols = struct.unpack("HHHH", x)[:2]
	return rows, cols

def setwinsize(r,c):
	# Assume ws_xpixel and ws_ypixel are zero.
	s = struct.pack("HHHH", r,c,0,0)
	x = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCSWINSZ, s)
print 'stdin tty:', os.ttyname(0)
print 'stdout tty:', os.ttyname(1)
print 'controlling terminal:', os.ctermid() 
print 'winsize %d,%d' % getwinsize()
print 'ENDTEST'
