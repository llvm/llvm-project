#!/usr/bin/env python
'''
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

import time
import pexpect
import sys

def getProcessResults(cmd, timeLimit=20):
  '''
  executes 'cmd' as a child process and returns the child's output,
  the duration of execution, and the process exit status. Aborts if
  child process does not generate output for 'timeLimit' seconds.
  '''
  output = ""
  startTime = time.time()
  child = pexpect.spawn(cmd, timeout=10)
  child.logfile = sys.stdout

  while 1:
    try:
      # read_nonblocking will add to 'outout' one byte at a time
      # newlines can show up as '\r\n' so we kill any '\r's which
      # will mess up the formatting for the viewer
      output += child.read_nonblocking(timeout=timeLimit).replace("\r","")
    except pexpect.EOF as e:
      print(str(e))
      # process terminated normally
      break
    except pexpect.TIMEOUT as e:
      print(str(e))
      output += "\nProcess aborted by FlashTest after %s seconds.\n" % timeLimit
      print(child.isalive())
      child.kill(9)
      break

  endTime = time.time()
  child.close(force=True)

  duration = endTime - startTime
  exitStatus = child.exitstatus

  return (output, duration, exitStatus)

cmd = "./ticker.py"

result, duration, exitStatus = getProcessResults(cmd)

print("result: %s" % result)
print("duration: %s" % duration)
print("exit-status: %s" % exitStatus)

