"""
Test "memory tag read" command on AArch64 Linux with MTE.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxMTEMemoryTagReadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_tag_read(self):
        if not self.isAArch64MTE():
            self.skipTest('Target must support MTE.')

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.c",
            line_number('main.c', '// Breakpoint here'),
            num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped',
                     'stop reason = breakpoint'])

        # Argument validation
        self.expect("memory tag read",
                substrs=["error: wrong number of arguments; expected at least <address-expression>, "
                         "at most <address-expression> <end-address-expression>"],
                error=True)
        self.expect("memory tag read buf buf+16 32",
                substrs=["error: wrong number of arguments; expected at least <address-expression>, "
                         "at most <address-expression> <end-address-expression>"],
                error=True)
        self.expect("memory tag read not_a_symbol",
                substrs=["error: Invalid address expression, address expression \"not_a_symbol\" "
                         "evaluation failed"],
                error=True)
        self.expect("memory tag read buf not_a_symbol",
                substrs=["error: Invalid end address expression, address expression \"not_a_symbol\" "
                         "evaluation failed"],
                error=True)
        # Inverted range
        self.expect("memory tag read buf buf-16",
                patterns=["error: End address \(0x[A-Fa-f0-9]+\) must be "
                          "greater than the start address \(0x[A-Fa-f0-9]+\)"],
                error=True)
        # Range of length 0
        self.expect("memory tag read buf buf",
                patterns=["error: End address \(0x[A-Fa-f0-9]+\) must be "
                          "greater than the start address \(0x[A-Fa-f0-9]+\)"],
                error=True)


        # Can't read from a region without tagging
        self.expect("memory tag read non_mte_buf",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+10 is not "
                         "in a memory tagged region"],
                error=True)

        # If there's no end address we assume 1 granule
        self.expect("memory tag read buf",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

        # Range of <1 granule is rounded up to 1 granule
        self.expect("memory tag read buf buf+8",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

        # Start address is aligned down, end aligned up
        self.expect("memory tag read buf+8 buf+24",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x1$"])

        # You may read up to the end of the tagged region
        # Layout is buf (MTE), buf2 (MTE), <unmapped/non MTE>
        # so we read from the end of buf2 here.
        self.expect("memory tag read buf2+page_size-16 buf2+page_size",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+, 0x[0-9A-Fa-f]+\): 0x0$"])

        # Ranges with any part outside the region will error
        self.expect("memory tag read buf2+page_size-16 buf2+page_size+32",
                patterns=["error: Address range 0x[0-9A-fa-f]+f0:0x[0-9A-Fa-f]+20 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag read buf2+page_size",
                patterns=["error: Address range 0x[0-9A-fa-f]+00:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)

        # You can read a range that spans more than one mapping
        # This spills into buf2 which is also MTE
        self.expect("memory tag read buf+page_size-16 buf+page_size+16",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+f0, 0x[0-9A-Fa-f]+00\): 0xf\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

        # Tags in start/end are ignored when creating the range.
        # So this is not an error despite start/end having different tags
        self.expect("memory tag read buf buf_alt_tag+16 ",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])
