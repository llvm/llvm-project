#!/usr/bin/env python

"""
Run gdb to disassemble a function, feed the bytes to 'llvm-mc -disassemble' command,
and display the disassembly result.

"""

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

def do_llvm_mc_disassembly(gdb_commands, gdb_options, exe, func, mc, mc_options):
    from cStringIO import StringIO 
    import pexpect

    gdb_prompt = "\r\n\(gdb\) "
    gdb = pexpect.spawn(('gdb %s' % gdb_options) if gdb_options else 'gdb')
    # Turn on logging for what gdb sends back.
    gdb.logfile_read = sys.stdout
    gdb.expect(gdb_prompt)

    # See if there any extra command(s) to execute before we issue the file command.
    for cmd in gdb_commands:
        gdb.sendline(cmd)
        gdb.expect(gdb_prompt)

    # Now issue the file command.
    gdb.sendline('file %s' % exe)
    gdb.expect(gdb_prompt)

    # Send the disassemble command.
    gdb.sendline('disassemble %s' % func)
    gdb.expect(gdb_prompt)

    # Get the output from gdb.
    gdb_output = gdb.before

    # Use StringIO to record the memory dump as well as the gdb assembler code.
    mc_input = StringIO()

    # These keep track of the states of our simple gdb_output parser.
    prev_line = None
    prev_addr = None
    curr_addr = None
    addr_diff = 0
    looking = False
    for line in gdb_output.split(os.linesep):
        if line.startswith('Dump of assembler code'):
            looking = True
            continue

        if line.startswith('End of assembler dump.'):
            looking = False
            prev_addr = curr_addr
            if mc_options and mc_options.find('arm') != -1:
                addr_diff = 4
            if mc_options and mc_options.find('thumb') != -1:
                # It is obviously wrong to assume the last instruction of the
                # function has two bytes.
                # FIXME
                addr_diff = 2

        if looking and line.startswith('0x'):
            # It's an assembler code dump.
            prev_addr = curr_addr
            curr_addr = line.split(None, 1)[0]
            if prev_addr and curr_addr:
                addr_diff = int(curr_addr, 16) - int(prev_addr, 16)

        if prev_addr and addr_diff > 0:
            # Feed the examining memory command to gdb.
            gdb.sendline('x /%db %s' % (addr_diff, prev_addr))
            gdb.expect(gdb_prompt)
            x_output = gdb.before
            # Get the last output line from the gdb examine memory command,
            # split the string into a 3-tuple with separator '>:' to handle
            # objc method names.
            memory_dump = x_output.split(os.linesep)[-1].partition('>:')[2].strip()
            #print "\nbytes:", memory_dump
            disasm_str = prev_line.partition('>:')[2]
            print >> mc_input, '%s # %s' % (memory_dump, disasm_str)

        # We're done with the processing.  Assign the current line to be prev_line.
        prev_line = line

    # Close the gdb session now that we are done with it.
    gdb.sendline('quit')
    gdb.expect(pexpect.EOF)
    gdb.close()

    # Write the memory dump into a file.
    with open('disasm-input.txt', 'w') as f:
        f.write(mc_input.getvalue())

    mc_cmd = '%s -disassemble %s disasm-input.txt' % (mc, mc_options)
    print "\nExecuting command:", mc_cmd
    os.system(mc_cmd)

    # And invoke llvm-mc with the just recorded file.
    #mc = pexpect.spawn('%s -disassemble %s disasm-input.txt' % (mc, mc_options))
    #mc.logfile_read = sys.stdout
    #print "mc:", mc
    #mc.close()
    

def main():
    # This is to set up the Python path to include the pexpect-2.4 dir.
    # Remember to update this when/if things change.
    scriptPath = sys.path[0]
    sys.path.append(os.path.join(scriptPath, os.pardir, os.pardir, 'test', 'pexpect-2.4'))

    parser = OptionParser(usage="""\
Run gdb to disassemble a function, feed the bytes to 'llvm-mc -disassemble' command,
and display the disassembly result.

Usage: %prog [options]
""")
    parser.add_option('-C', '--gdb-command',
                      type='string', action='append', metavar='COMMAND',
                      default=[], dest='gdb_commands',
                      help='Command(s) gdb executes after starting up (can be empty)')
    parser.add_option('-O', '--gdb-options',
                      type='string', action='store',
                      dest='gdb_options',
                      help="""The options passed to 'gdb' command if specified.""")
    parser.add_option('-e', '--executable',
                      type='string', action='store',
                      dest='executable',
                      help="""The executable to do disassembly on.""")
    parser.add_option('-f', '--function',
                      type='string', action='store',
                      dest='function',
                      help="""The function name (could be an address to gdb) for disassembly.""")
    parser.add_option('-m', '--llvm-mc',
                      type='string', action='store',
                      dest='llvm_mc',
                      help="""The llvm-mc executable full path, if specified.
                      Otherwise, it must be present in your PATH environment.""")

    parser.add_option('-o', '--options',
                      type='string', action='store',
                      dest='llvm_mc_options',
                      help="""The options passed to 'llvm-mc -disassemble' command if specified.""")

    opts, args = parser.parse_args()

    gdb_commands = opts.gdb_commands
    gdb_options = opts.gdb_options

    if not opts.executable:
        parser.print_help()
        sys.exit(1)
    executable = opts.executable

    if not opts.function:
        parser.print_help()
        sys.exit(1)
    function = opts.function

    llvm_mc = opts.llvm_mc if opts.llvm_mc else which('llvm-mc')
    if not llvm_mc:
        parser.print_help()
        sys.exit(1)

    # This is optional.  For example:
    # --options='-triple=arm-apple-darwin -debug-only=arm-disassembler'
    llvm_mc_options = opts.llvm_mc_options

    # We have parsed the options.
    print "gdb commands:", gdb_commands
    print "gdb options:", gdb_options
    print "executable:", executable
    print "function:", function
    print "llvm-mc:", llvm_mc
    print "llvm-mc options:", llvm_mc_options

    do_llvm_mc_disassembly(gdb_commands, gdb_options, executable, function, llvm_mc, llvm_mc_options)

if __name__ == '__main__':
    main()
