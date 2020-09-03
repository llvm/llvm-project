"""
Test that variables with unsigned types display correctly.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class UnsignedTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        """Test that variables with unsigned types display correctly."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// Set break point at this line", lldb.SBFileSpec("main.cpp"))

        # Test that unsigned types display correctly.
        self.expect(
            "frame variable --show-types --no-args",
            VARIABLES_DISPLAYED_CORRECTLY,
            patterns=["\((short unsigned int|unsigned short)\) the_unsigned_short = 99"],
            substrs=[
                "(unsigned char) the_unsigned_char = 'c'",
                "(unsigned int) the_unsigned_int = 99",
                "(unsigned long) the_unsigned_long = 99",
                "(unsigned long long) the_unsigned_long_long = 99",
                "(uint32_t) the_uint32 = 99"])
