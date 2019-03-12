#!/usr/bin/env python
"""This script can only be killed by SIGKILL."""
import signal, time

# Ignore interrupt, hangup and continue signals - only SIGKILL will work
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
signal.signal(signal.SIGCONT, signal.SIG_IGN)

print('READY')
while True:
    time.sleep(10)
    