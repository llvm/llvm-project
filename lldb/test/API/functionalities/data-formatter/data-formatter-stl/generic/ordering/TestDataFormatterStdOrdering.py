"""
Test std::*_ordering summary.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdOrderingTestCase(TestBase):
    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "frame variable",
            substrs=[
                "(std::partial_ordering) po_less = less",
                "(std::partial_ordering) po_equivalent = equivalent",
                "(std::partial_ordering) po_greater = greater",
                "(std::partial_ordering) po_unordered = unordered",
                "(std::weak_ordering) wo_less = less",
                "(std::weak_ordering) wo_equivalent = equivalent",
                "(std::weak_ordering) wo_greater = greater",
                "(std::strong_ordering) so_less = less",
                "(std::strong_ordering) so_equal = equal",
                "(std::strong_ordering) so_equivalent = equal",
                "(std::strong_ordering) so_greater = greater",
            ],
        )

        frame = self.frame()
        self.assertEqual(frame.FindVariable("po_less").summary, "less")
        self.assertEqual(frame.FindVariable("po_equivalent").summary, "equivalent")
        self.assertEqual(frame.FindVariable("po_greater").summary, "greater")
        self.assertEqual(frame.FindVariable("po_unordered").summary, "unordered")
        self.assertEqual(frame.FindVariable("wo_less").summary, "less")
        self.assertEqual(frame.FindVariable("wo_equivalent").summary, "equivalent")
        self.assertEqual(frame.FindVariable("wo_greater").summary, "greater")
        self.assertEqual(frame.FindVariable("so_less").summary, "less")
        self.assertEqual(frame.FindVariable("so_equal").summary, "equal")
        self.assertEqual(frame.FindVariable("so_equivalent").summary, "equal")
        self.assertEqual(frame.FindVariable("so_greater").summary, "greater")

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()
