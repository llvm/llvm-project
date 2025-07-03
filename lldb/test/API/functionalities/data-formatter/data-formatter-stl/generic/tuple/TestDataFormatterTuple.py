"""
Test lldb data formatter subsystem.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDataFormatterStdTuple(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.line = line_number("main.cpp", "// break here")
        self.namespace = "std"

    def do_test(self):
        """Test that std::tuple is displayed correctly"""
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        tuple_name = self.namespace + "::tuple"
        self.expect("frame variable empty", substrs=[tuple_name, "size=0", "{}"])

        self.expect(
            "frame variable one_elt",
            substrs=[tuple_name, "size=1", "{", "[0] = 47", "}"],
        )

        self.expect(
            "frame variable three_elts",
            substrs=[
                tuple_name,
                "size=3",
                "{",
                "[0] = 1",
                "[1] = 47",
                '[2] = "foo"',
                "}",
            ],
        )

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()
