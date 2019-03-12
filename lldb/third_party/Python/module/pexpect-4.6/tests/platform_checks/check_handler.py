#!/usr/bin/env python
import signal
import os
import time
import pty
import sys
import fcntl
import tty
GLOBAL_SIGCHLD_RECEIVED = 0
                                                                                 
def nonblock (fd):                                                           
	# if O_NDELAY is set read() returns 0 (ambiguous with EOF).
	# if O_NONBLOCK is set read() returns -1 and sets errno to EAGAIN
	original_flags = fcntl.fcntl (fd, fcntl.F_GETFL, 0)
	flags = original_flags | os.O_NONBLOCK
	fcntl.fcntl(fd, fcntl.F_SETFL, flags)
	return original_flags

def signal_handler (signum, frame):
	print '<HANDLER>'
	global GLOBAL_SIGCHLD_RECEIVED
	status = os.waitpid (-1, os.WNOHANG)
	if status[0] == 0:
		print 'No process for waitpid:', status
	else:
		print 'Status:', status
	print 'WIFEXITED(status):', os.WIFEXITED(status[1])
	print 'WEXITSTATUS(status):', os.WEXITSTATUS(status[1]) 
	GLOBAL_SIGCHLD_RECEIVED = 1

def main ():
	signal.signal (signal.SIGCHLD, signal_handler)
	pid, fd = pty.fork()
	if pid == 0:
		os.write (sys.stdout.fileno(), 'This is a test.\nThis is a test.')
		time.sleep(10000)
	nonblock (fd)
	tty.setraw(fd) #STDIN_FILENO)
	print 'Sending SIGKILL to child pid:', pid
	time.sleep(2)
	os.kill (pid, signal.SIGKILL)

	print 'Entering to sleep...'
	try:
		time.sleep(2)
	except:
		print 'Sleep interrupted'
	try:
		os.kill(pid, 0)
		print '\tChild is alive. This is ambiguous because it may be a Zombie.'
	except OSError as e:
		print '\tChild appears to be dead.'
#		print str(e)
	print
	print 'Reading from master fd:', os.read (fd, 1000)



if __name__ == '__main__':
	main ()
