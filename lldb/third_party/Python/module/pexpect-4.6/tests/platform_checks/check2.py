#!/usr/bin/env python
import signal
import os
import time

def signal_handler (signum, frame):
	print 'Signal handler called with signal:', signum
	print 'signal.SIGCHLD=', signal.SIGKILL

# Create a child process for us to kill.
pid = os.fork()
if pid == 0:
	time.sleep(10000)

#signal.signal (signal.SIGCHLD, signal.SIG_IGN)
signal.signal (signal.SIGCHLD, signal_handler)

print 'Sending SIGKILL to child pid:', pid
os.kill (pid, signal.SIGKILL)

# SIGCHLD should interrupt sleep.
# Note that this is a race.
# It is possible that the signal handler will get called
# before we try to sleep, but this has not happened yet.
# But in that case we can only tell by order of printed output.
interrupted = 0
try:
	time.sleep(10)
except:
	print 'sleep was interrupted by signal.'
	interrupted = 1

if not interrupted:
	print 'ERROR. Signal did not interrupt sleep.'
else:
	print 'Signal interrupted sleep. This is good.'

# Let's see if the process is alive.
try:
	os.kill(pid, 0)
	print 'Child is alive. This is ambiguous because it may be a Zombie.'
except OSError as e:
	print 'Child appears to be dead.'

