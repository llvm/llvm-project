#!/usr/bin/env python

"""
Run a program via lldb until it fails.
The lldb executable is located via your PATH env variable, if not specified.
"""

from __future__ import print_function

import os
import sys
from optparse import OptionParser


def is_exe(fpath):
    """Check whether fpath is an executable."""
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    """Find the full path to a program, or return None."""
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def do_lldb_launch_loop(lldb_command, exe, exe_options):
    import pexpect
    import time

    prompt = "\(lldb\) "
    lldb = pexpect.spawn(lldb_command)
    # Turn on logging for what lldb sends back.
    lldb.logfile_read = sys.stdout
    lldb.expect(prompt)

    # Now issue the file command.
    # print "sending 'file %s' command..." % exe
    lldb.sendline('file %s' % exe)
    lldb.expect(prompt)

    # Loop until it faults....
    count = 0
    # while True:
    #    count = count + 1
    for i in range(100):
        count = i
        # print "sending 'process launch -- %s' command... (iteration: %d)" %
        # (exe_options, count)
        lldb.sendline('process launch -- %s' % exe_options)
        index = lldb.expect(['Process .* exited with status',
                             'Process .* stopped',
                             pexpect.TIMEOUT])
        if index == 0:
            # We'll try again later.
            time.sleep(3)
        elif index == 1:
            # Perfect, our process had stopped; break out of the loop.
            break
        elif index == 2:
            # Something went wrong.
            print("TIMEOUT occurred:", str(lldb))

    # Give control of lldb shell to the user.
    lldb.interact()


def main():
    # This is to set up the Python path to include the pexpect-2.4 dir.
    # Remember to update this when/if things change.
    scriptPath = sys.path[0]
    sys.path.append(
        os.path.join(
            scriptPath,
            os.pardir,
            os.pardir,
            'test',
            'pexpect-2.4'))

    parser = OptionParser(usage="""\
%prog [options]
Run a program via lldb until it fails.
The lldb executable is located via your PATH env variable, if not specified.\
""")
    parser.add_option('-l', '--lldb-command',
                      type='string', action='store', metavar='LLDB_COMMAND',
                      default='lldb', dest='lldb_command',
                      help='Full path to your lldb command')
    parser.add_option(
        '-e',
        '--executable',
        type='string',
        action='store',
        dest='exe',
        help="""(Mandatory) The executable to launch via lldb.""")
    parser.add_option(
        '-o',
        '--options',
        type='string',
        action='store',
        default='',
        dest='exe_options',
        help="""The args/options passed to the launched program, if specified.""")

    opts, args = parser.parse_args()

    lldb_command = which(opts.lldb_command)

    if not opts.exe:
        parser.print_help()
        sys.exit(1)
    exe = opts.exe

    exe_options = opts.exe_options

    # We have parsed the options.
    print("lldb command:", lldb_command)
    print("executable:", exe)
    print("executable options:", exe_options)

    do_lldb_launch_loop(lldb_command, exe, exe_options)

if __name__ == '__main__':
    main()
