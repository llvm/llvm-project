"""
Test "memory tag read" and "memory tag write" commands
on AArch64 Linux with MTE.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxMTEMemoryTagAccessTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setup_mte_test(self):
        if not self.isAArch64MTE():
            self.skipTest('Target must support MTE.')

        # Required to check that commands remove non-address bits
        # other than the memory tags.
        if not self.isAArch64PAuth():
            self.skipTest('Target must support pointer authentication')

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
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0 \(mismatch\)$"])

        # Range of <1 granule is rounded up to 1 granule
        self.expect("memory tag read mte_buf mte_buf+8",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0 \(mismatch\)$"])

        # Start address is aligned down, end aligned up
        self.expect("memory tag read mte_buf+8 mte_buf+24",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x1 \(mismatch\)$"])

        # You may read up to the end of the tagged region
        # Layout is mte_buf, mte_buf_2, non_mte_buf.
        # So we read from the end of mte_buf_2 here.
        self.expect("memory tag read mte_buf_2+page_size-16 mte_buf_2+page_size",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+, 0x[0-9A-Fa-f]+\): 0x0$"])

        # Ranges with any part outside the region will error
        self.expect("memory tag read mte_buf_2+page_size-16 mte_buf_2+page_size+32",
                patterns=["error: Address range 0x[0-9A-Fa-f]+f0:0x[0-9A-Fa-f]+20 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag read mte_buf_2+page_size",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)

        # You can read a range that spans more than one mapping
        # This spills into mte_buf2 which is also MTE
        self.expect("memory tag read mte_buf+page_size-16 mte_buf+page_size+16",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+f0, 0x[0-9A-Fa-f]+00\): 0xf \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0 \(mismatch\)$"])

        # Top byte is ignored when creating the range, not just the 4 tag bits.
        # So even though these two pointers have different top bytes
        # and the start's is > the end's, this is not an error.
        self.expect("memory tag read mte_buf_alt_tag mte_buf+16",
                patterns=["Logical tag: 0xa\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x0 \(mismatch\)$"])

        # Mismatched tags are marked. The logical tag is taken from the start address.
        self.expect("memory tag read mte_buf+(8*16) mte_buf_alt_tag+(8*16)+48",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+80, 0x[0-9A-Fa-f]+90\): 0x8 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+90, 0x[0-9A-Fa-f]+a0\): 0x9\n"
                          "\[0x[0-9A-Fa-f]+a0, 0x[0-9A-Fa-f]+b0\): 0xa \(mismatch\)$"])

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
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x1 \(mismatch\)$"])

        # You can write multiple tags, range calculated for you
        self.expect("memory tag write mte_buf 10 11 12")
        self.expect("memory tag read mte_buf mte_buf+48",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0xa \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0xb \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+20, 0x[0-9A-Fa-f]+30\): 0xc \(mismatch\)$"])

        # You may write up to the end of a tagged region
        # (mte_buf_2's intial tags will all be 0)
        self.expect("memory tag write mte_buf_2+page_size-16 0xe")
        self.expect("memory tag read mte_buf_2+page_size-16 mte_buf_2+page_size",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+, 0x[0-9A-Fa-f]+\): 0xe \(mismatch\)$"])

        # Ranges with any part outside the region will error
        self.expect("memory tag write mte_buf_2+page_size-16 6 7",
                patterns=["error: Address range 0x[0-9A-Fa-f]+f0:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag write mte_buf_2+page_size 6",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)
        self.expect("memory tag write mte_buf_2+page_size 6 7 8",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+30 "
                          "is not in a memory tagged region"],
                error=True)

        # You can write to a range that spans two mappings, as long
        # as they are both tagged.
        # buf and buf2 are next to each other so this wirtes into buf2.
        self.expect("memory tag write mte_buf+page_size-16 1 2")
        self.expect("memory tag read mte_buf+page_size-16 mte_buf+page_size+16",
                patterns=["Logical tag: 0x9\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+f0, 0x[0-9A-Fa-f]+00\): 0x1 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x2 \(mismatch\)$"])

        # Even if a page is read only the debugger can still write to it
        self.expect("memory tag write mte_read_only 1")
        self.expect("memory tag read mte_read_only",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x1 \(mismatch\)$"])

        # Trying to write a value > maximum tag value is an error
        self.expect("memory tag write mte_buf 99",
                patterns=["error: Found tag 0x63 which is > max MTE tag value of 0xf."],
                error=True)

        # You can provide an end address and have lldb repeat the tags as needed
        # The range is checked in the same way it is for "memory tag read"
        self.expect("memory tag write mte_buf 9 -e",
                patterns=["error: last option requires an argument"],
                error=True)
        self.expect("memory tag write mte_buf 9 -e food",
                patterns=["error: address expression \"food\" evaluation failed"],
                error=True)
        self.expect("memory tag write mte_buf_2 9 --end-addr mte_buf_2",
                patterns=["error: End address \(0x[A-Fa-f0-9]+\) must be "
                          "greater than the start address \(0x[A-Fa-f0-9]+\)"],
                error=True)
        self.expect("memory tag write mte_buf_2 9 --end-addr mte_buf_2-16",
                patterns=["error: End address \(0x[A-Fa-f0-9]+\) must be "
                          "greater than the start address \(0x[A-Fa-f0-9]+\)"],
                error=True)
        self.expect("memory tag write mte_buf_2 9 --end-addr mte_buf_2+page_size+16",
                patterns=["error: Address range 0x[0-9A-Fa-f]+00:0x[0-9A-Fa-f]+10 "
                          "is not in a memory tagged region"],
                error=True)

        # Tags are repeated across the range
        # For these we'll read one extra to make sure we don't over write
        self.expect("memory tag write mte_buf_2 4 5 --end-addr mte_buf_2+48")
        self.expect("memory tag read mte_buf_2 mte_buf_2+64",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x4 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x5 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+20, 0x[0-9A-Fa-f]+30\): 0x4 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+30, 0x[0-9A-Fa-f]+40\): 0x0$"])

        # Since this aligns like tag read does, the start is aligned down and the end up.
        # Meaning that start/end tells you the start/end granule that will be written.
        # This matters particularly if either are misaligned.

        # Here start moves down so the final range is mte_buf_2 -> mte_buf_2+32
        self.expect("memory tag write mte_buf_2+8 6 -end-addr mte_buf_2+32")
        self.expect("memory tag read mte_buf_2 mte_buf_2+48",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x6 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x6 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+20, 0x[0-9A-Fa-f]+30\): 0x4 \(mismatch\)$"])

        # If we do the same with a misaligned end, it also moves but upward.
        # The intial range is 2 granules but the final range is mte_buf_2 -> mte_buf_2+48
        self.expect("memory tag write mte_buf_2+8 3 -end-addr mte_buf_2+32+8")
        self.expect("memory tag read mte_buf_2 mte_buf_2+64",
                patterns=["Logical tag: 0x0\n"
                          "Allocation tags:\n"
                          "\[0x[0-9A-Fa-f]+00, 0x[0-9A-Fa-f]+10\): 0x3 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+10, 0x[0-9A-Fa-f]+20\): 0x3 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+20, 0x[0-9A-Fa-f]+30\): 0x3 \(mismatch\)\n"
                          "\[0x[0-9A-Fa-f]+30, 0x[0-9A-Fa-f]+40\): 0x0$"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_memory_read_tag_display(self):
        self.setup_mte_test()

        # Reading from an untagged range should not be any different
        self.expect("memory read non_mte_buf non_mte_buf+16",
                substrs=["tag"], matching=False)

        # show-tags option is required
        self.expect("memory read mte_buf mte_buf+32 -f \"x\" -l 1 -s 16",
                patterns=["tag"], matching=False)

        # Reading 16 bytes per line means 1 granule and so 1 tag per line
        self.expect("memory read mte_buf mte_buf+32 -f \"x\" -l 1 -s 16 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+00: 0x0+ \(tag: 0x0\)\n"
                    "0x[0-9A-Fa-f]+10: 0x0+ \(tag: 0x1\)"
                    ])

        # If bytes per line is > granule size then you get multiple tags
        # per line.
        self.expect("memory read mte_buf mte_buf+32 -f \"x\" -l 1 -s 32 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+00: 0x0+ \(tags: 0x0 0x1\)\n"
                    ])

        # Reading half a granule still shows you the tag for that granule
        self.expect("memory read mte_buf mte_buf+8 -f \"x\" -l 1 -s 8 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+00: 0x0+ \(tag: 0x0\)\n"
                    ])

        # We can read a whole number of granules but split them over more lines
        # than there are granules. Tags are shown repeated for each applicable line.
        self.expect("memory read mte_buf+32 mte_buf+64 -f \"x\" -l 1 -s 8 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+20: 0x0+ \(tag: 0x2\)\n"
                    "0x[0-9A-Fa-f]+28: 0x0+ \(tag: 0x2\)\n"
                    "0x[0-9A-Fa-f]+30: 0x0+ \(tag: 0x3\)\n"
                    "0x[0-9A-Fa-f]+38: 0x0+ \(tag: 0x3\)"
                    ])

        # Also works if we misalign the start address. Note the first tag is shown
        # only once here and we have a new tag on the last line.
        # (bytes per line == the misalignment here)
        self.expect("memory read mte_buf+32+8 mte_buf+64+8 -f \"x\" -l 1 -s 8 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+28: 0x0+ \(tag: 0x2\)\n"
                    "0x[0-9A-Fa-f]+30: 0x0+ \(tag: 0x3\)\n"
                    "0x[0-9A-Fa-f]+38: 0x0+ \(tag: 0x3\)\n"
                    "0x[0-9A-Fa-f]+40: 0x0+ \(tag: 0x4\)"
                    ])

        # We can do the same thing but where the misaligment isn't equal to
        # bytes per line. This time, some lines cover multiple granules and
        # so show multiple tags.
        self.expect("memory read mte_buf+32+4 mte_buf+64+4 -f \"x\" -l 1 -s 8 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+24: 0x0+ \(tag: 0x2\)\n"
                    "0x[0-9A-Fa-f]+2c: 0x0+ \(tags: 0x2 0x3\)\n"
                    "0x[0-9A-Fa-f]+34: 0x0+ \(tag: 0x3\)\n"
                    "0x[0-9A-Fa-f]+3c: 0x0+ \(tags: 0x3 0x4\)"
                    ])

        # If you read a range that includes non tagged areas those areas
        # simply aren't annotated.

        # Initial part of range is untagged
        self.expect("memory read mte_buf-16 mte_buf+32 -f \"x\" -l 1 -s 16 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+f0: 0x0+\n"
                    "0x[0-9A-Fa-f]+00: 0x0+ \(tag: 0x0\)\n"
                    "0x[0-9A-Fa-f]+10: 0x0+ \(tag: 0x1\)"
                    ])

        # End of range is untagged
        self.expect("memory read mte_buf+page_size-16 mte_buf+page_size+16 -f \"x\" -l 1 -s 16 --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+f0: 0x0+ \(tag: 0xf\)\n"
                    "0x[0-9A-Fa-f]+00: 0x0+"
                    ])

        # The smallest MTE range we can get is a single page so we just check
        # parts of this result. Where we read from before the tagged page to after it.
        # Add --force here because we're reading just over 4k.
        self.expect(
                "memory read mte_read_only-16 mte_read_only+page_size+16 -f \"x\" -l 1 -s 16 --force --show-tags",
                patterns=[
                    "0x[0-9A-Fa-f]+f0: 0x0+\n"
                    "0x[0-9A-Fa-f]+00: 0x0+ \(tag: 0x0\)\n",
                    "0x[0-9A-Fa-f]+f0: 0x0+ \(tag: 0x0\)\n"
                    "0x[0-9A-Fa-f]+00: 0x0+"
                    ])

        # Some parts of a line might be tagged and others untagged.
        # <no tag> is shown in where the tag would be, to keep the order intact.
        self.expect("memory read mte_buf-16 mte_buf+32 -f \"x\" -l 1 -s 32 --show-tags",
                patterns=["0x[0-9A-Fa-f]+f0: 0x0+ \(tags: <no tag> 0x0\)"])
        self.expect(
                "memory read mte_read_only+page_size-16 mte_read_only+page_size+16 -f \"x\" -l 1 -s 32  --show-tags",
                patterns=["0x[0-9A-Fa-f]+f0: 0x0+ \(tags: 0x0 <no tag>\)"])

        # Here the start address is unaligned so we cover 3 granules instead of 2
        self.expect("memory read mte_buf-16+4 mte_buf+32+4 -f \"x\" -l 1 -s 32 --show-tags",
                patterns=["0x[0-9A-Fa-f]+f4: 0x0+ \(tags: <no tag> 0x0 0x1\)"])
        self.expect(
                "memory read mte_read_only+page_size-16+4 mte_read_only+page_size+16+4 -f \"x\" -l 1 -s 32 --show-tags",
                patterns=["0x[0-9A-Fa-f]+f4: 0x0+ \(tags: 0x0 <no tag> <no tag>\)"])

        # Some formats call DumpDataExtractor multiple times,
        # check that those print tags only once per line.
        self.expect("memory read mte_buf mte_buf+32 -f \"x\" --show-tags",
                patterns=["0x[0-9A-Fa-f]+00: 0x0+ 0x0+ 0x0+ 0x0+ \(tag: 0x0\)\n",
                          "0x[0-9A-Fa-f]+10: 0x0+ 0x0+ 0x0+ 0x0+ \(tag: 0x1\)"])

        self.expect("memory read mte_buf mte_buf+32 -f \"bytes with ASCII\" --show-tags",
                patterns=["0x[0-9A-Fa-f]+00: (00 ){16} \.{16} \(tag: 0x0\)\n",
                          "0x[0-9A-Fa-f]+10: (00 ){16} \.{16} \(tag: 0x1\)"])

        self.expect("memory read mte_buf mte_buf+32 -f \"uint8_t[]\" -s 16 -l 1 --show-tags",
                patterns=["0x[0-9A-Fa-f]+00: \{(0x00 ){15}0x00\} \(tag: 0x0\)\n"
                          "0x[0-9A-Fa-f]+10: \{(0x00 ){15}0x00\} \(tag: 0x1\)"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_memory_read_tag_display_repeated(self):
        """Test that the --show-tags option is kept when repeating the memory read command."""
        self.setup_mte_test()

        self.expect("memory read mte_buf mte_buf+16 -f \"x\" --show-tags",
                    patterns=["0x[0-9A-fa-f]+00: 0x0+ 0x0+ 0x0+ 0x0+ \(tag: 0x0\)"])
        # Equivalent to just pressing enter on the command line.
        self.expect("memory read",
                    patterns=["0x[0-9A-fa-f]+10: 0x0+ 0x0+ 0x0+ 0x0+ \(tag: 0x1\)"])

        # You can add the argument to an existing repetition without resetting
        # the whole command. Though all other optional arguments will reset to
        # their default values when you do this.
        self.expect("memory read mte_buf mte_buf+16 -f \"x\"",
                    patterns=["0x[0-9A-fa-f]+00: 0x0+ 0x0+ 0x0+ 0x0+"])
        self.expect("memory read",
                    patterns=["0x[0-9A-fa-f]+10: 0x0+ 0x0+ 0x0+ 0x0+"])
        # Note that the formatting returns to default here.
        self.expect("memory read --show-tags",
                    patterns=["0x[0-9A-fa-f]+20: (00 )+ \.+ \(tag: 0x2\)"])
        self.expect("memory read",
                    patterns=["0x[0-9A-fa-f]+30: (00 )+ \.+ \(tag: 0x3\)"])

        # A fresh command reverts to the default of tags being off.
        self.expect("memory read mte_buf mte_buf+16 -f \"x\"",
                    patterns=["0x[0-9A-fa-f]+00: 0x0+ 0x0+ 0x0+ 0x0+"])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    @skipUnlessAArch64MTELinuxCompiler
    def test_mte_memory_find(self):
        """Test the --show-tags option with memory find."""
        self.setup_mte_test()

        # No result, nothing changes.
        self.expect("memory find -s \"foo\" mte_buf mte_buf+32 --show-tags",
            substrs=["data not found within the range."])

        cmd = "memory find -s \"LLDB\" mte_buf+64 mte_buf+512"
        found_pattern = "data found at location: 0x[0-9A-Fa-f]+80"
        results_patterns = [
            "0x[0-9A-Fa-f]+80: 4c 4c 44 42 (00 )+ LLDB\.+",
            "0x[0-9A-Fa-f]+90: 00 00 00 00 (00 )+ \.+"
        ]

        # Default is not to show tags
        self.expect(cmd, patterns=[found_pattern, *results_patterns])
        self.expect(cmd + " --show-tags", patterns=[found_pattern,
                    results_patterns[0] + " \(tag: 0x8\)",
                    results_patterns[1] + " \(tag: 0x9\)"])

        # Uses the same logic as memory read to handle misalignment.
        self.expect("memory find -s \"DB\" mte_buf+64 mte_buf+512 --show-tags",
            patterns=[
                "data found at location: 0x[0-9A-Fa-f]+82\n"
                "0x[0-9A-Fa-f]+82: 44 42 (00 )+ DB\.+ \(tags: 0x8 0x9\)\n",
                "0x[0-9A-Fa-f]+92: 00 00 (00 )+ ..\.+ \(tags: 0x9 0xa\)"])
