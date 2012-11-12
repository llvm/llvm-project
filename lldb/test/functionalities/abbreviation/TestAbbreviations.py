"""
Test some lldb command abbreviations.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class AbbreviationsTestCase(TestBase):

    mydir = os.path.join("functionalities", "abbreviation")

    def test_nonrunning_command_abbreviations (self):
        self.expect("ap script",
                    startstr = "The following commands may relate to 'script':",
                    substrs = ['breakpoint command add',
                               'breakpoint command list',
                               'breakpoint list',
                               'command alias',
                               'expression',
                               'script'])

        self.runCmd("com a alias com al")
        self.runCmd("alias gurp help")
        self.expect("gurp target create",
                    substrs = ['Syntax: target create <cmd-options> <filename>'])
        self.runCmd("com u gurp")
        self.expect("gurp",
                    COMMAND_FAILED_AS_EXPECTED, error = True,
                    substrs = ["error: 'gurp' is not a valid command."])

        # Only one matching command: execute it.
        self.expect("h",
                    startstr = "The following is a list of built-in, permanent debugger commands:")

        # Execute cleanup function during test tear down
        def cleanup():
            self.runCmd("command alias t thread select")
        self.addTearDownHook(cleanup)

        # Several matching commands: list them and error out.
        self.runCmd("command unalias t")
        self.expect("t",
                    COMMAND_FAILED_AS_EXPECTED, error = True,
                    substrs = ["Ambiguous command 't'. Possible matches:",
                               "target", "thread", "type"])

        self.expect("com sou ./change_prompt.lldb",
                    patterns = ["Executing commands in '.*change_prompt.lldb'"])

        self.expect("settings show prompt",
                    startstr = 'prompt (string) = "[with-three-trailing-spaces]   "')


        self.runCmd("settings clear prompt")
        self.expect("settings show prompt",
                    startstr = 'prompt (string) = "(lldb) "')


        self.expect("lo li",
                    startstr = "Logging categories for ")

        self.runCmd("se se prompt Sycamore> ")
        self.expect("se sh prompt",
                    startstr = 'prompt (string) = "Sycamore> "')

        self.runCmd("se cl prompt")
        self.expect("set sh prompt",
                    startstr = 'prompt (string) = "(lldb) "')

        # We don't want to display the stdout if not in TraceOn() mode.
        if not self.TraceOn():
            self.HideStdout()

        self.runCmd (r'''sc print "\n\n\tHello!\n"''')


    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym ()
        self.running_abbreviations ()

    @dwarf_test
    def test_with_dwarf (self):
        self.buildDwarf ()
        self.running_abbreviations ()

    def running_abbreviations (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("fil " + exe,
                    patterns = [ "Current executable set to .*a.out.*" ])

        # By default, the setting interpreter.expand-regex-aliases is false.
        self.expect("_regexp-br product", matching=False,
                    substrs = [ "breakpoint set --name" ])

        match_object = lldbutil.run_break_set_command (self, "br s -n sum")
        lldbutil.check_breakpoint_result (self, match_object, symbol_name='sum', symbol_match_exact=False, num_locations=1)

        match_object = lldbutil.run_break_set_command (self, "br s -f main.cpp -l 32")
        lldbutil.check_breakpoint_result (self, match_object, file_name='main.cpp', line_number=32, num_locations=1)

        self.runCmd("br co a -s python 1 -o 'print frame'")
        self.expect("br co l 1",
                    substrs = [ "Breakpoint 1:",
                                "Breakpoint commands:",
                                "print frame" ])

        self.runCmd("br co del 1")
        self.expect("breakpoint command list 1",
                    startstr = "Breakpoint 1 does not have an associated command.")

        self.expect("br di",
                    startstr = 'All breakpoints disabled. (3 breakpoints)')

        self.expect("bre e",
                    startstr = "All breakpoints enabled. (3 breakpoints)")

        self.expect("break list",
                    substrs = ["1: name = 'product', locations = 1",
                               "2: name = 'sum', locations = 1",
                               "3: file ='main.cpp', line = 32, locations = 1"])
        self.expect("br cl -l 32 -f main.cpp",
                    startstr = "1 breakpoints cleared:",
                    substrs = ["3: file ='main.cpp', line = 32, locations = 1"])

        # Add a future to terminate the current process being debugged.
        #
        # The test framework relies on detecting either "run" or "process launch"
        # command to automatically kill the inferior upon tear down.
        # But we'll be using "pro la" command to launch the inferior.
        self.addTearDownHook(lambda: self.runCmd("process kill"))
        self.expect("pro la",
                    patterns = [ "Process .* launched: "])

        self.expect("pro st",
                    patterns = [ "Process .* stopped",
                                 "thread #1:",
                                 "a.out",
                                 "sum\(a=1238, b=78392\)",
                                 "at main.cpp\:25",
                                 "stop reason = breakpoint 2.1" ])

        # ARCH, if not specified, defaults to x86_64.
        if self.getArchitecture() in ["", 'x86_64', 'i386']:
            self.expect("dis -f",
                        startstr = "a.out`sum(int, int)",
                        substrs = [' push',
                                   ' mov',
                                   ' addl ',
                                   'ret'],
                        patterns = ['(leave|popq|popl)'])

        self.expect("i d l main.cpp",
                    patterns = ["Line table for .*main.cpp in `a.out"])

        self.expect("i d se",
                    patterns = ["Dumping sections for [0-9]+ modules."])

        self.expect("i d symf",
                    patterns = ["Dumping debug symbols for [0-9]+ modules."])

        self.expect("i d symt",
                    patterns = ["Dumping symbol table for [0-9]+ modules."])

        if sys.platform.startswith("darwin"):
            self.expect("i li",
                        substrs = [ 'a.out',
                                    '/usr/lib/dyld',
                                    '/usr/lib/libstdc++',
                                    '/usr/lib/libSystem.B.dylib'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

