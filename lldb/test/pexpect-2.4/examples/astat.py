#!/usr/bin/env python

"""This runs Apache Status on the remote host and returns the number of requests per second.

./astat.py [-s server_hostname] [-u username] [-p password]
    -s : hostname of the remote server to login to.
    -u : username to user for login.
    -p : Password to user for login.

Example:
    This will print information about the given host:
        ./astat.py -s www.example.com -u mylogin -p mypassword

"""

import os, sys, time, re, getopt, getpass
import traceback
import pexpect, pxssh

def exit_with_usage():

    print globals()['__doc__']
    os._exit(1)

def main():

    ######################################################################
    ## Parse the options, arguments, get ready, etc.
    ######################################################################
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h?s:u:p:', ['help','h','?'])
    except Exception, e:
        print str(e)
        exit_with_usage()
    options = dict(optlist)
    if len(args) > 1:
        exit_with_usage()

    if [elem for elem in options if elem in ['-h','--h','-?','--?','--help']]:
        print "Help:"
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
    p.expect('([0-9]+\.[0-9]+)\s*requests/sec')
    requests_per_second = p.match.groups()[0]
    p.logout()
    print requests_per_second

if __name__ == "__main__":
    try:
        main()
    except Exception, e:
        print str(e)
        traceback.print_exc()
        os._exit(1)

