# coding=utf8
"""
Test std::u8string summary with MSVC's STL.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlU8StringDataFormatterTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["msvcstl"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "frame variable",
            substrs=[
                '(std::u8string) u8_string_small = u8"🍄"',
                '(std::u8string) u8_string = u8"❤️👍📄📁😃🧑‍🌾"',
                '(std::u8string) u8_empty = u8""',
                '(std::u8string) u8_text = u8"ABC"',
            ],
        )
