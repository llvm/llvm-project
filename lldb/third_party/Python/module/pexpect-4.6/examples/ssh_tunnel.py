#!/usr/bin/env python

'''This starts an SSH tunnel to a given host. If the SSH process ever dies then
this script will detect that and restart it. I use this under Cygwin to keep
open encrypted tunnels to port 25 (SMTP), port 143 (IMAP4), and port 110
(POP3). I set my mail client to talk to localhost and I keep this script
running in the background.

Note that this is a rather stupid script at the moment because it just looks to
see if any ssh process is running. It should really make sure that our specific
ssh process is running. The problem is that ssh is missing a very useful
feature. It has no way to report the process id of the background daemon that
it creates with the -f command. This would be a really useful script if I could
figure a way around this problem.

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

from __future__ import print_function

from __future__ import absolute_import

import pexpect
import getpass
import time


try:
    raw_input
except NameError:
    raw_input = input


# SMTP:25 IMAP4:143 POP3:110
tunnel_command = 'ssh -C -N -f -L 25:127.0.0.1:25 -L 143:127.0.0.1:143 -L 110:127.0.0.1:110 %(user)@%(host)'
host = raw_input('Hostname: ')
user = raw_input('Username: ')
X = getpass.getpass('Password: ')

def get_process_info ():

    # This seems to work on both Linux and BSD, but should otherwise be considered highly UNportable.

    ps = pexpect.run ('ps ax -O ppid')
    pass

def start_tunnel ():

    try:
        ssh_tunnel = pexpect.spawn (tunnel_command % globals())
        ssh_tunnel.expect ('password:')
        time.sleep (0.1)
        ssh_tunnel.sendline (X)
        time.sleep (60) # Cygwin is slow to update process status.
        ssh_tunnel.expect (pexpect.EOF)

    except Exception as e:
        print(str(e))

def main ():

    while True:
        ps = pexpect.spawn ('ps')
        time.sleep (1)
        index = ps.expect (['/usr/bin/ssh', pexpect.EOF, pexpect.TIMEOUT])
        if index == 2:
            print('TIMEOUT in ps command...')
            print(str(ps))
            time.sleep (13)
        if index == 1:
            print(time.asctime(), end=' ')
            print('restarting tunnel')
            start_tunnel ()
            time.sleep (11)
            print('tunnel OK')
        else:
            # print 'tunnel OK'
            time.sleep (7)

if __name__ == '__main__':

    main ()

# This was for older SSH versions that didn't have -f option
#tunnel_command = 'ssh -C -n -L 25:%(host)s:25 -L 110:%(host)s:110 %(user)s@%(host)s -f nothing.sh'
#nothing_script = '''#!/bin/sh
#while true; do sleep 53; done
#'''

