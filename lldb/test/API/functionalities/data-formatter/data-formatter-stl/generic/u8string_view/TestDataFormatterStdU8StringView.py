# coding=utf8
"""
Test std::u8string_view summary.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdU8StringViewDataFormatterTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        string_view_name = (
            "std::basic_string_view<char8_t, std::char_traits<char8_t>>"
            if self.getDebugInfo() == "pdb"
            else "std::u8string_view"
        )

        self.expect(
            "frame variable",
            substrs=[
                f'({string_view_name}) u8_string_small = u8"🍄"',
                f'({string_view_name}) u8_string = u8"❤️👍📄📁😃🧑‍🌾"',
                f'({string_view_name}) u8_empty = u8""',
                f'({string_view_name}) u8_text = u8"ABCd"',
            ],
        )

    @expectedFailureAll(bugnumber="No libc++ formatters for std::u8string_view yet.")
    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @expectedFailureAll(bugnumber="No libstdc++ formatters for std::u8string_view yet.")
    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvc(self):
        self.build()
        self.do_test()
