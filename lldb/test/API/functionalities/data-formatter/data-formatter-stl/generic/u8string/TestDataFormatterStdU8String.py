# coding=utf8
"""
Test std::u8string summary.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdU8StringDataFormatterTestCase(TestBase):
    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "frame variable",
            substrs=[
                '(std::u8string) u8_string_small = u8"🍄"',
                '(std::u8string) u8_string = u8"❤️👍📄📁😃🧑‍🌾"',
                '(std::u8string) u8_empty = u8""',
                '(std::u8string) u8_text = u8"ABCd"',
            ],
        )

    @expectedFailureAll(bugnumber="No libc++ formatters for std::u8string yet.")
    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @expectedFailureAll(bugnumber="No libstdc++ formatters for std::u8string yet.")
    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvc(self):
        self.build()
        self.do_test()
