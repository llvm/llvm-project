#!/usr/bin/env python
import signal
import os
import time
import pty
import sys
GLOBAL_SIGCHLD_RECEIVED = 0

def signal_handler (signum, frame):
    print '<HANDLER>'
    global GLOBAL_SIGCHLD_RECEIVED
    status = os.waitpid (-1, os.WNOHANG)
    print 'WIFEXITED(status):', os.WIFEXITED(status)
    print 'WEXITSTATUS(status):', os.WEXITSTATUS(status) 
    GLOBAL_SIGCHLD_RECEIVED = 1

def main ():
#	sig_test ('SIG_IGN', 'ptyfork', 'yes')
	sig_test ('handler', 'ptyfork', 'yes')
#	sig_test ('SIG_IGN', 'ptyfork', 'no')
#	sig_test ('handler', 'ptyfork', 'no')
#	sig_test ('SIG_IGN', 'osfork', 'yes')
#	sig_test ('handler', 'osfork', 'yes')
#	sig_test ('SIG_IGN', 'osfork', 'no')
#	sig_test ('handler', 'osfork', 'no')

def sig_test (sig_handler_type, fork_type, child_output):
	print 'Testing with:'
	print '\tsig_handler_type:', sig_handler_type
	print '\tfork_type:', fork_type
	print '\tchild_output:', child_output

	if sig_handler_type == 'SIG_IGN':
		signal.signal (signal.SIGCHLD, signal.SIG_IGN)
	else:
		signal.signal (signal.SIGCHLD, signal_handler)
	pid = -1
	fd = -1
	if fork_type == 'ptyfork':
		pid, fd = pty.fork()
	else:
		pid = os.fork()

	if pid == 0:
		if child_output == 'yes':
			os.write (sys.stdout.fileno(), 'This is a test.\nThis is a test.')
		time.sleep(10000)

	#print 'Sending SIGKILL to child pid:', pid
	time.sleep(2)
	os.kill (pid, signal.SIGKILL)

	#print 'Entering to sleep...'
	try:
		time.sleep(2)
	except:
		pass
	try:
		os.kill(pid, 0)
		print '\tChild is alive. This is ambiguous because it may be a Zombie.'
	except OSError as e:
		print '\tChild appears to be dead.'
#		print str(e)
	print

if __name__ == '__main__':
	main ()
