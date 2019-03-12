#!/usr/bin/env python
import signal
import os
import time
import pty

def signal_handler (signum, frame):
    print 'Signal handler called with signal:', signum
    print 'signal.SIGCHLD=', signal.SIGKILL

# First thing we do is set up a handler for SIGCHLD.
signal.signal (signal.SIGCHLD, signal.SIG_IGN)

print 'PART 1 -- Test signal handling with empty pipe.'
# Create a child process for us to kill.
try:
    pid, fd = pty.fork()
except Exception as e:
    print str(e)

if pid == 0:
#    os.write (sys.stdout.fileno(), 'This is a test.\n This is a test.')
    time.sleep(10000)

print 'Sending SIGKILL to child pid:', pid
os.kill (pid, signal.SIGKILL)

# SIGCHLD should interrupt sleep.
# Note that this is a race.
# It is possible that the signal handler will get called
# before we try to sleep, but this has not happened yet.
# But in that case we can only tell by order of printed output.
print 'Entering sleep...'
try:
    time.sleep(10)
except:
    print 'sleep was interrupted by signal.'

# Just for fun let's see if the process is alive.
try:
    os.kill(pid, 0)
    print 'Child is alive. This is ambiguous because it may be a Zombie.'
except OSError as e:
    print 'Child appears to be dead.'

print 'PART 2 -- Test signal handling with full pipe.'
# Create a child process for us to kill.
try:
    pid, fd = pty.fork()
except Exception as e:
    print str(e)

if pid == 0:
    os.write (sys.stdout.fileno(), 'This is a test.\n This is a test.')
    time.sleep(10000)

print 'Sending SIGKILL to child pid:', pid
os.kill (pid, signal.SIGKILL)

# SIGCHLD should interrupt sleep.
# Note that this is a race.
# It is possible that the signal handler will get called
# before we try to sleep, but this has not happened yet.
# But in that case we can only tell by order of printed output.
print 'Entering sleep...'
try:
    time.sleep(10)
except:
    print 'sleep was interrupted by signal.'

# Just for fun let's see if the process is alive.
try:
    os.kill(pid, 0)
    print 'Child is alive. This is ambiguous because it may be a Zombie.'
except OSError as e:
    print 'Child appears to be dead.'

