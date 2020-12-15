"""Show bitfields and check that they display correctly."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CppBitfieldsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    # BitFields exhibit crashes in record layout on Windows
    # (http://llvm.org/pr21800)
    @skipIfWindows
    def test_and_run_command(self):
        """Test 'frame variable ...' on a variable with bitfields."""
        self.build()

        lldbutil.run_to_source_breakpoint(self, '// Set break point at this line.',
          lldb.SBFileSpec("main.cpp", False))

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        self.expect("expr (lba.a)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['unsigned int', '2'])
        self.expect("expr (lbb.b)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['unsigned int', '3'])
        self.expect("expr (lbc.c)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['unsigned int', '4'])
        self.expect("expr (lbd.a)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['unsigned int', '5'])
        self.expect("expr (clang_example.f.a)", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['uint64_t', '1'])

        self.expect("expr uwbf",
            substrs=['a = 255',
                    'b = 65535',
                    'c = 4294967295',
                    'x = 4294967295'] )

        self.expect("expr uwubf",
            substrs=['a = 16777215',
                    'x = 4294967295'] )

        self.expect(
            "frame variable --show-types lba",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(int:32)  = ',
                '(unsigned int:20) a = 2',
                ])

        self.expect(
            "frame variable --show-types lbb",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(unsigned int:1) a = 1',
                '(int:31)  =',
                '(unsigned int:20) b = 3',
                ])

        self.expect(
            "frame variable --show-types lbc",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(int:22)  =',
                '(unsigned int:1) a = 1',
                '(unsigned int:1) b = 0',
                '(unsigned int:5) c = 4',
                '(unsigned int:1) d = 1',
                '(int:2)  =',
                '(unsigned int:20) e = 20',
                ])

        self.expect(
            "frame variable --show-types lbd",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(char [3]) arr = "ab"',
                '(int:32)  =',
                '(unsigned int:20) a = 5',
                ])

        self.expect(
            "frame variable --show-types clang_example",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                   '(int:22)  =',
                   '(uint64_t:1) a = 1',
                   '(uint64_t:1) b = 0',
                   '(uint64_t:1) c = 1',
                   '(uint64_t:1) d = 0',
                   '(uint64_t:1) e = 1',
                   '(uint64_t:1) f = 0',
                   '(uint64_t:1) g = 1',
                   '(uint64_t:1) h = 0',
                   '(uint64_t:1) i = 1',
                   '(uint64_t:1) j = 0',
                   '(uint64_t:1) k = 1',
                ])

        self.expect(
            "frame variable --show-types derived",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                '(uint32_t) b_a = 2',
                '(uint32_t:1) d_a = 1',
                ])
