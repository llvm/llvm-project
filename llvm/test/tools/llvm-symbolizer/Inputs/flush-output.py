from __future__ import print_function
import os
import subprocess
import sys
import threading


def kill_subprocess(process):
    process.kill()
    os._exit(1)


# Pass -f=none and --output-style=GNU to get only one line of output per input.
cmd = subprocess.Popen(
    [sys.argv[1], "--obj=" + sys.argv[2], "-f=none", "--output-style=GNU"],
    stdout=subprocess.PIPE,
    stdin=subprocess.PIPE,
)
watchdog = threading.Timer(20, kill_subprocess, args=[cmd])
watchdog.start()
cmd.stdin.write(b"0\n")
cmd.stdin.flush()
print(cmd.stdout.readline())
cmd.stdin.write(b"bad\n")
cmd.stdin.flush()
print(cmd.stdout.readline())
watchdog.cancel()
