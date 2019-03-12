#!/usr/bin/env python

'''This runs Apache Status on the remote host and returns the number of requests per second.

./astat.py [-s server_hostname] [-u username] [-p password]
    -s : hostname of the remote server to login to.
    -u : username to user for login.
    -p : Password to user for login.

Example:
    This will print information about the given host:
        ./astat.py -s www.example.com -u mylogin -p mypassword

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

import os
import sys
import getopt
import getpass
import pxssh


try:
    raw_input
except NameError:
    raw_input = input


def exit_with_usage():

    print(globals()['__doc__'])
    os._exit(1)


def main():

    ######################################################################
    ## Parse the options, arguments, get ready, etc.
    ######################################################################
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h?s:u:p:', ['help','h','?'])
    except Exception as e:
        print(str(e))
        exit_with_usage()
    options = dict(optlist)
    if len(args) > 1:
        exit_with_usage()

    if [elem for elem in options if elem in ['-h','--h','-?','--?','--help']]:
        print("Help:")
        exit_with_usage()

    if '-s' in options:
        hostname = options['-s']
    else:
        hostname = raw_input('hostname: ')
    if '-u' in options:
        username = options['-u']
    else:
        username = raw_input('username: ')
    if '-p' in options:
        password = options['-p']
    else:
        password = getpass.getpass('password: ')

    #
    # Login via SSH
    #
    p = pxssh.pxssh()
    p.login(hostname, username, password)
    p.sendline('apachectl status')
    p.expect(r'([0-9]+\.[0-9]+)\s*requests/sec')
    requests_per_second = p.match.groups()[0]
    p.logout()
    print(requests_per_second)

if __name__ == "__main__":
    main()
