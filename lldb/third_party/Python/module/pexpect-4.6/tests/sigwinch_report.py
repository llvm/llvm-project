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
from  __future__ import print_function

import signal, time, struct, fcntl, termios, sys

def getwinsize():
    '''This returns the window size of the child tty.
    The return value is a tuple of (rows, cols).
    '''
    if 'TIOCGWINSZ' in dir(termios):
        TIOCGWINSZ = termios.TIOCGWINSZ
    else:
        TIOCGWINSZ = 1074295912 # Assume
    s = struct.pack('HHHH', 0, 0, 0, 0)
    x = fcntl.ioctl(sys.stdout.fileno(), TIOCGWINSZ, s)
    return struct.unpack('HHHH', x)[0:2]

def handler(signum, frame):
    print('signal')
    sys.stdout.flush()
    print('SIGWINCH:', getwinsize ())
    sys.stdout.flush()

print("Initial Size:", getwinsize())
print("setting handler for SIGWINCH")
signal.signal(signal.SIGWINCH, handler)
print("READY")

while 1:
    sys.stdout.flush()
    time.sleep(1)

