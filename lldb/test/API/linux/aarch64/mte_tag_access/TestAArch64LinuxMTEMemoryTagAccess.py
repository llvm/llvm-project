"""
Test "memory tag read" and "memory tag write" commands
on AArch64 Linux with MTE.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxMTEMemoryTagAccessTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setup_mte_test(self):
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

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_tag_read(self):
        self.setup_mte_test()

        # Argument validation
        self.expect("memory tag read",
                substrs=["error: wrong number of arguments; expected at least <address-expression>, "
                         "at most <address-expression> <end-address-expression>"],
                error=True)
        self.expect("memory tag read mte_buf buf+16 32",
                substrs=["error: wrong number of arguments; expected at least <address-expression>, "
                         "at most <address-expression> <end-address-expression>"],
                error=True)
        self.expect("memory tag read not_a_symbol",
                substrs=["error: Invalid address expression, address expression \"not_a_symbol\" "
                         "evaluation failed"],
                error=True)
        self.expect("memory tag read mte_buf not_a_symbol",
                substrs=["error: Invalid end address expression, address expression \"not_a_symbol\" "
                         "evaluation failed"],
                error=True)
        # Inverted range
        self.expect("memory tag read mte_buf mte_buf-16",
                patterns=["error: End address \(0x[A-Fa-f0-9]+\) must be "
                          "greater than the start address \(0x[A-Fa-f0-9]+\)"],
                error=True)
        # Range of length 0
        self.expect("memory tag read mte_buf mte_buf",
                patterns=["error: End address \(0x[A-Fa-f0-9]+\) must be "
                          "greater than the start address \(0x[A-Fa-f0-9]+\)"],
                error=True)


        # Can't read from a region without tagging
        self.expect("memory tag read non_mte_buf",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+10 is not "
                         "in a memory tagged region"],
                error=True)

        # If there's no end address we assume 1 granule
        self.expect("memory tag read mte_buf",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

        # Range of <1 granule is rounded up to 1 granule
        self.expect("memory tag read mte_buf mte_buf+8",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

        # Start address is aligned down, end aligned up
        self.expect("memory tag read mte_buf+8 mte_buf+24",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x1$"])

        # You may read up to the end of the tagged region
        # Layout is mte_buf, mte_buf_2, non_mte_buf.
        # So we read from the end of mte_buf_2 here.
        self.expect("memory tag read mte_buf_2+page_size-16 mte_buf_2+page_size",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+, 0x[0-9A-Fa-f]+\): 0x0$"])

        # Ranges with any part outside the region will error
        self.expect("memory tag read mte_buf_2+page_size-16 mte_buf_2+page_size+32",
                patterns=["error: Address range 0x[0-9A-fa-f]+f0:0x[0-9A-Fa-f]+20 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag read mte_buf_2+page_size",
                patterns=["error: Address range 0x[0-9A-fa-f]+00:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)

        # You can read a range that spans more than one mapping
        # This spills into mte_buf2 which is also MTE
        self.expect("memory tag read mte_buf+page_size-16 mte_buf+page_size+16",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+f0, 0x[0-9A-Fa-f]+00\): 0xf\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

        # Tags in start/end are ignored when creating the range.
        # So this is not an error despite start/end having different tags
        self.expect("memory tag read mte_buf mte_buf_alt_tag+16 ",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0$"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_tag_write(self):
        self.setup_mte_test()

        # Argument validation
        self.expect("memory tag write",
                substrs=[" wrong number of arguments; expected "
                         "<address-expression> <tag> [<tag> [...]]"],
                error=True)
        self.expect("memory tag write mte_buf",
                substrs=[" wrong number of arguments; expected "
                         "<address-expression> <tag> [<tag> [...]]"],
                error=True)
        self.expect("memory tag write not_a_symbol 9",
                substrs=["error: Invalid address expression, address expression \"not_a_symbol\" "
                         "evaluation failed"],
                error=True)

        # Can't write to a region without tagging
        self.expect("memory tag write non_mte_buf 9",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+10 is not "
                         "in a memory tagged region"],
                error=True)

        # Start address is aligned down so we write to the granule that contains it
        self.expect("memory tag write mte_buf+8 9")
        # Make sure we only modified the first granule
        self.expect("memory tag read mte_buf mte_buf+32",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x9\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x1$"])

        # You can write multiple tags, range calculated for you
        self.expect("memory tag write mte_buf 10 11 12")
        self.expect("memory tag read mte_buf mte_buf+48",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0xa\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0xb\n"
                          "\[0x[0-9A-Fa-f]+20, 0x[0-9A-Fa-f]+30\): 0xc$"])

        # You may write up to the end of a tagged region
        # (mte_buf_2's intial tags will all be 0)
        self.expect("memory tag write mte_buf_2+page_size-16 0xe")
        self.expect("memory tag read mte_buf_2+page_size-16 mte_buf_2+page_size",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+, 0x[0-9A-Fa-f]+\): 0xe$"])

        # Ranges with any part outside the region will error
        self.expect("memory tag write mte_buf_2+page_size-16 6 7",
                patterns=["error: Address range 0x[0-9A-fa-f]+f0:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag write mte_buf_2+page_size 6",
                patterns=["error: Address range 0x[0-9A-fa-f]+00:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag write mte_buf_2+page_size 6 7 8",
                patterns=["error: Address range 0x[0-9A-fa-f]+00:0x[0-9A-Fa-f]+30 "
                          "is not in a memory tagged region"],
                error=True)

        # You can write to a range that spans two mappings, as long
        # as they are both tagged.
        # buf and buf2 are next to each other so this wirtes into buf2.
        self.expect("memory tag write mte_buf+page_size-16 1 2")
        self.expect("memory tag read mte_buf+page_size-16 mte_buf+page_size+16",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+f0, 0x[0-9A-Fa-f]+00\): 0x1\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x2$"])

        # Even if a page is read only the debugger can still write to it
        self.expect("memory tag write mte_read_only 1")
        self.expect("memory tag read mte_read_only",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x1$"])

        # Trying to write a value > maximum tag value is an error
        self.expect("memory tag write mte_buf 99",
                patterns=["error: Found tag 0x63 which is > max MTE tag value of 0xf."],
                error=True)
