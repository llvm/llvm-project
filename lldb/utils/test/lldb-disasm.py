#!/usr/bin/env python

"""
Run lldb to disassemble all the available functions for an executable image.

"""

import os
import re
import sys
from optparse import OptionParser

def setupSysPath():
    """
    Add LLDB.framework/Resources/Python and the test dir to the sys.path.
    """
    # Get the directory containing the current script.
    scriptPath = sys.path[0]
    if not scriptPath.endswith(os.path.join('utils', 'test')):
        print "This script expects to reside in lldb's utils/test directory."
        sys.exit(-1)

    # This is our base name component.
    base = os.path.abspath(os.path.join(scriptPath, os.pardir, os.pardir))

    # This is for the goodies in the test directory under base.
    sys.path.append(os.path.join(base,'test'))

    # These are for xcode build directories.
    xcode3_build_dir = ['build']
    xcode4_build_dir = ['build', 'lldb', 'Build', 'Products']
    dbg = ['Debug']
    rel = ['Release']
    bai = ['BuildAndIntegration']
    python_resource_dir = ['LLDB.framework', 'Resources', 'Python']

    dbgPath  = os.path.join(base, *(xcode3_build_dir + dbg + python_resource_dir))
    dbgPath2 = os.path.join(base, *(xcode4_build_dir + dbg + python_resource_dir))
    relPath  = os.path.join(base, *(xcode3_build_dir + rel + python_resource_dir))
    relPath2 = os.path.join(base, *(xcode4_build_dir + rel + python_resource_dir))
    baiPath  = os.path.join(base, *(xcode3_build_dir + bai + python_resource_dir))
    baiPath2 = os.path.join(base, *(xcode4_build_dir + bai + python_resource_dir))

    lldbPath = None
    if os.path.isfile(os.path.join(dbgPath, 'lldb.py')):
        lldbPath = dbgPath
    elif os.path.isfile(os.path.join(dbgPath2, 'lldb.py')):
        lldbPath = dbgPath2
    elif os.path.isfile(os.path.join(relPath, 'lldb.py')):
        lldbPath = relPath
    elif os.path.isfile(os.path.join(relPath2, 'lldb.py')):
        lldbPath = relPath2
    elif os.path.isfile(os.path.join(baiPath, 'lldb.py')):
        lldbPath = baiPath
    elif os.path.isfile(os.path.join(baiPath2, 'lldb.py')):
        lldbPath = baiPath2

    if not lldbPath:
        print 'This script requires lldb.py to be in either ' + dbgPath + ',',
        print relPath + ', or ' + baiPath
        sys.exit(-1)

    # This is to locate the lldb.py module.  Insert it right after sys.path[0].
    sys.path[1:1] = [lldbPath]
    #print "sys.path:", sys.path


def run_command(ci, cmd, res, echo=True):
    if echo:
        print "run command:", cmd
    ci.HandleCommand(cmd, res)
    if res.Succeeded():
        if echo:
            print "run_command output:", res.GetOutput()
    else:
        if echo:
            print "run command failed!"
            print "run_command error:", res.GetError()

def do_lldb_disassembly(lldb_commands, exe, disassemble_options, num_symbols,
                        symbols_to_disassemble,
                        re_symbol_pattern,
                        quiet_disassembly):
    import lldb, atexit, re

    # Create the debugger instance now.
    dbg = lldb.SBDebugger.Create()
    if not dbg:
            raise Exception('Invalid debugger instance')

    # Register an exit callback.
    atexit.register(lambda: lldb.SBDebugger.Terminate())

    # We want our debugger to be synchronous.
    dbg.SetAsync(False)

    # Get the command interpreter from the debugger.
    ci = dbg.GetCommandInterpreter()
    if not ci:
        raise Exception('Could not get the command interpreter')

    # And the associated result object.
    res = lldb.SBCommandReturnObject()

    # See if there any extra command(s) to execute before we issue the file command.
    for cmd in lldb_commands:
        run_command(ci, cmd, res, not quiet_disassembly)

    # Now issue the file command.
    run_command(ci, 'file %s' % exe, res, not quiet_disassembly)

    # Create a target.
    #target = dbg.CreateTarget(exe)
    target = dbg.GetSelectedTarget()
    stream = lldb.SBStream()

    def IsCodeType(symbol):
        """Check whether an SBSymbol represents code."""
        return symbol.GetType() == lldb.eSymbolTypeCode

    # Define a generator for the symbols to disassemble.
    def symbol_iter(num, symbols, re_symbol_pattern, target, verbose):
        # If we specify the symbols to disassemble, ignore symbol table dump.
        if symbols:
            for i in range(len(symbols)):
                if verbose:
                    print "symbol:", symbols[i]
                yield symbols[i]
        else:
            limited = True if num != -1 else False
            if limited:
                count = 0
            if re_symbol_pattern:
                pattern = re.compile(re_symbol_pattern)
            stream = lldb.SBStream()
            for m in target.module_iter():
                if verbose:
                    print "module:", m
                for s in m:
                    if limited and count >= num:
                        return
                    # If a regexp symbol pattern is supplied, consult it.
                    if re_symbol_pattern:
                        # If the pattern does not match, look for the next symbol.
                        if not pattern.match(s.GetName()):
                            continue

                    # If we come here, we're ready to disassemble the symbol.
                    if verbose:
                        print "symbol:", s.GetName()
                    if IsCodeType(s):
                        if limited:
                            count = count + 1
                            if verbose:
                                print "returning symbol:", s.GetName()
                        yield s.GetName()
                    if verbose:
                        print "start address:", s.GetStartAddress()
                        print "end address:", s.GetEndAddress()
                        s.GetDescription(stream)
                        print "symbol description:", stream.GetData()
                        stream.Clear()

    # Disassembly time.
    for symbol in symbol_iter(num_symbols, symbols_to_disassemble, re_symbol_pattern, target, not quiet_disassembly):
        cmd = "disassemble %s '%s'" % (disassemble_options, symbol)
        run_command(ci, cmd, res, not quiet_disassembly)


def main():
    # This is to set up the Python path to include the pexpect-2.4 dir.
    # Remember to update this when/if things change.
    scriptPath = sys.path[0]
    sys.path.append(os.path.join(scriptPath, os.pardir, os.pardir, 'test', 'pexpect-2.4'))

    parser = OptionParser(usage="""\
Run lldb to disassemble all the available functions for an executable image.

Usage: %prog [options]
""")
    parser.add_option('-C', '--lldb-command',
                      type='string', action='append', metavar='COMMAND',
                      default=[], dest='lldb_commands',
                      help='Command(s) lldb executes after starting up (can be empty)')
    parser.add_option('-e', '--executable',
                      type='string', action='store',
                      dest='executable',
                      help="""Mandatory: the executable to do disassembly on.""")
    parser.add_option('-o', '--options',
                      type='string', action='store',
                      dest='disassemble_options',
                      help="""Mandatory: the options passed to lldb's 'disassemble' command.""")
    parser.add_option('-q', '--quiet-disassembly',
                      action='store_true', default=False,
                      dest='quiet_disassembly',
                      help="""The symbol(s) to invoke lldb's 'disassemble' command on, if specified.""")
    parser.add_option('-n', '--num-symbols',
                      type='int', action='store', default=-1,
                      dest='num_symbols',
                      help="""The number of symbols to disassemble, if specified.""")
    parser.add_option('-p', '--symbol_pattern',
                      type='string', action='store',
                      dest='re_symbol_pattern',
                      help="""The regular expression of symbols to invoke lldb's 'disassemble' command.""")
    parser.add_option('-s', '--symbol',
                      type='string', action='append', metavar='SYMBOL', default=[],
                      dest='symbols_to_disassemble',
                      help="""The symbol(s) to invoke lldb's 'disassemble' command on, if specified.""")
    
    opts, args = parser.parse_args()

    lldb_commands = opts.lldb_commands

    if not opts.executable or not opts.disassemble_options:
        parser.print_help()
        sys.exit(1)

    executable = opts.executable
    disassemble_options = opts.disassemble_options
    quiet_disassembly = opts.quiet_disassembly
    num_symbols = opts.num_symbols
    symbols_to_disassemble = opts.symbols_to_disassemble
    re_symbol_pattern = opts.re_symbol_pattern

    # We have parsed the options.
    if not quiet_disassembly:
        print "lldb commands:", lldb_commands
        print "executable:", executable
        print "disassemble options:", disassemble_options
        print "quiet disassembly output:", quiet_disassembly
        print "num of symbols to disassemble:", num_symbols
        print "symbols to disassemble:", symbols_to_disassemble
        print "regular expression of symbols to disassemble:", re_symbol_pattern

    setupSysPath()
    do_lldb_disassembly(lldb_commands, executable, disassemble_options,
                        num_symbols,
                        symbols_to_disassemble,
                        re_symbol_pattern,
                        quiet_disassembly)

if __name__ == '__main__':
    main()
