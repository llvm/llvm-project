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
import termios
import sys

# a dumb PAM will print the password prompt first then set ECHO
# False. What it should do it set ECHO False first then print the
# prompt. Otherwise, if we see the password prompt and type out
# password real fast before it turns off ECHO then some or all of
# our password might be visibly echod back to us. Sounds unlikely?
# It happens.

print("fake password:")
sys.stdout.flush()
time.sleep(3)
attr = termios.tcgetattr(sys.stdout)
attr[3] = attr[3] & ~termios.ECHO
termios.tcsetattr(sys.stdout, termios.TCSANOW, attr)
time.sleep(12)
attr[3] = attr[3] | termios.ECHO
termios.tcsetattr(sys.stdout, termios.TCSANOW, attr)
time.sleep(2)
