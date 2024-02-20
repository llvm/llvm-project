"""
Test that "memory read" and "memory find" remove non address bits from
address arguments.

These tests use the top byte ignore feature of AArch64. Which Linux
always enables.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxTaggedMemoryReadTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setup_test(self):
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self,
            "main.c",
            line_number("main.c", "// Set break point at this line."),
            num_expected_locations=1,
        )

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_tagged_memory_read(self):
        self.setup_test()

        # If we do not remove non address bits, this can fail in two ways.
        # 1. We attempt to read much more than 16 bytes, probably more than
        #    the default 1024 byte read size. Which will error.
        # 2. We error because end address is < start address since end's
        #    tag is < start's tag.
        #
        # Each time we check that the printed line addresses do not include
        # either of the tags we set. Those bits are a property of the
        # pointer not of the memory it points to.
        tagged_addr_pattern = "0x(34|46)[0-9A-Fa-f]{14}:.*"
        self.expect(
            "memory read ptr1 ptr2+16", patterns=[tagged_addr_pattern], matching=False
        )
        # Check that the stored previous end address is stripped
        self.expect("memory read", patterns=[tagged_addr_pattern], matching=False)
        # Would fail if we don't remove non address bits because 0x56... > 0x34...
        self.expect(
            "memory read ptr2 ptr1+16", patterns=[tagged_addr_pattern], matching=False
        )
        self.expect("memory read", patterns=[tagged_addr_pattern], matching=False)

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_tagged_memory_find(self):
        self.setup_test()

        # If memory find doesn't remove non-address bits one of two
        # things happen.
        # 1. It tries to search a gigantic amount of memory.
        #    We're not going to test for this because a failure
        #    would take a very long time and perhaps even find the
        #    target value randomly.
        # 2. It thinks high address <= low address, which we check below.

        self.runCmd("memory find -s '?' ptr2 ptr1+32")

        self.assertTrue(self.res.Succeeded())
        out = self.res.GetOutput()
        # memory find does not fail when it doesn't find the data.
        # First check we actually got something.
        self.assertRegex(out, "data found at location: 0x[0-9A-Fa-f]+")
        # Then that the location found does not display the tag bits.
        self.assertNotRegexpMatches(
            out, "data found at location: 0x(34|56)[0-9A-Fa-f]+"
        )
