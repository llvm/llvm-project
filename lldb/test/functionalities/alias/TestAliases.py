"""
Test lldb command aliases.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class AliasTestCase(TestBase):

    mydir = os.path.join("functionalities", "alias")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym (self):
        self.buildDsym ()
        self.alias_tests ()

    @dwarf_test
    def test_with_dwarf (self):
        self.buildDwarf ()
        self.alias_tests ()

    def alias_tests (self):
        exe = os.path.join (os.getcwd(), "a.out")
        self.expect("file " + exe,
                    patterns = [ "Current executable set to .*a.out" ])


        def cleanup():
            self.runCmd('command unalias hello', check=False)
            self.runCmd('command unalias python', check=False)
            self.runCmd('command unalias pp', check=False)
            self.runCmd('command unalias alias', check=False)
            self.runCmd('command unalias unalias', check=False)
            self.runCmd('command unalias myrun', check=False)
            self.runCmd('command unalias bp', check=False)
            self.runCmd('command unalias bpa', check=False)
            self.runCmd('command unalias bpi', check=False)
            self.runCmd('command unalias bfl', check=False)
            self.runCmd('command unalias exprf', check=False)
            self.runCmd('command unalias exprf2', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd (r'''command alias hello expr (int) printf ("\n\nHello, anybody!\n\n")''')

        self.runCmd ("command alias python script")

        # We don't want to display the stdout if not in TraceOn() mode.
        if not self.TraceOn():
            self.HideStdout()

        self.runCmd (r'''python print "\n\n\nWhoopee!\n\n\n"''')
#        self.expect (r'''python print "\n\n\nWhoopee!\n\n\n"''',
#                     substrs = [ "Whoopee!" ])

        self.runCmd (r'''python print "\n\t\x68\x65\x6c\x6c\x6f\n"''')
#        self.expect (r'''python print "\n\t\x68\x65\x6c\x6c\x6f\n"''',
#                     substrs = [ "hello" ])

        self.runCmd (r'''command alias pp python print "\n\t\x68\x65\x6c\x6c\x6f\n"''')
        self.runCmd ("pp")
#        self.expect ("pp",
#                     substrs = [ "hello" ])


        self.runCmd ("command alias alias command alias")
        self.runCmd ("command alias unalias command unalias")

        self.runCmd ("alias myrun process launch -t %1 --")
        self.runCmd ("alias bp breakpoint")

        self.expect ("alias bpa bp add",
                     COMMAND_FAILED_AS_EXPECTED, error = True,
                     substrs = [ "'add' is not a valid sub-command of 'bp'" ])

        self.runCmd ("alias bpa bp command add")
        self.runCmd ("alias bpi bp list")

        break_results = lldbutil.run_break_set_command (self, "bp set -n foo")
        lldbutil.check_breakpoint_result (self, break_results, num_locations=1, symbol_name='foo', symbol_match_exact=False)

        break_results = lldbutil.run_break_set_command (self, "bp set -n sum")
        lldbutil.check_breakpoint_result (self, break_results, num_locations=1, symbol_name='sum', symbol_match_exact=False)

        self.runCmd ("alias bfl bp set -f %1 -l %2")

        break_results = lldbutil.run_break_set_command (self, "bfl main.cpp 32")
        lldbutil.check_breakpoint_result (self, break_results, num_locations=1, file_name='main.cpp', line_number=32)

        self.expect ("bpi",
                     startstr = "Current breakpoints:",
                     substrs = [ "1: name = 'foo', locations = 1",
                                 "2: name = 'sum', locations = 1",
                                 "3: file ='main.cpp', line = 32, locations = 1" ])

        self.runCmd ("bpa -s python 1 -o 'print frame; print bp_loc'")
        self.runCmd ("bpa -s command 2 -o 'frame variable b'")
        self.expect ("bpi -f",
                     substrs = [ "Current breakpoints:",
                                 "1: name = 'foo', locations = 1",
                                 "print frame; print bp_loc",
                                 "2: name = 'sum', locations = 1",
                                 "frame variable b" ])


        self.expect ("help run",
                     substrs = [ "'run' is an abbreviation for 'process launch -c /bin/bash --'" ])

        self.expect ("help -a run",
                     substrs = [ "'run' is an abbreviation for 'process launch -c /bin/bash --'" ])

        self.expect ("help -a",
                     substrs = [ 'run', 'process launch -c /bin/bash' ])

        self.expect ("help", matching=False,
                     substrs = [ "'run'", 'process launch -c /bin/bash' ])

        self.expect ("run",
                     patterns = [ "Process .* launched: .*a.out" ])

        self.expect (r'''expression (int) printf("\x68\x65\x6c\x6c\x6f\n")''',
                     substrs = [ "(int) $",
                                 "= 6" ])

        self.expect ("hello",
                     substrs = [ "(int) $", 
                                 "= 19" ])

        self.expect ("expr -f x -- 68",
                     substrs = [ "(int) $",
                                 "= 0x00000044" ])

        self.runCmd ("alias exprf expr -f %1")
        self.runCmd ("alias exprf2 expr --raw -f %1 --")
        self.expect ("exprf x -- 1234",
                     substrs = [ "(int) $",
                                 "= 0x000004d2" ])

        self.expect ('exprf2 c "Hi there!"',
                     substrs = [ "[0] = 'H'",
                                 "[1] = 'i'",
                                 "[2] = ' '",
                                 "[3] = 't'",
                                 "[4] = 'h'",
                                 "[5] = 'e'",
                                 "[6] = 'r'",
                                 "[7] = 'e'",
                                 "[8] = '!'",
                                 "[9] = '\\0'" ])
        

        self.expect ("exprf x 1234",
                     COMMAND_FAILED_AS_EXPECTED, error = True,
                     substrs = [ "use of undeclared identifier 'f'",
                                 "1 errors parsing expression" ])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()

