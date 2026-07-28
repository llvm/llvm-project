"""
Test that the expression path of a payload enum's case includes the name of
the enum-typed member it was projected from.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEnumExpressionPath(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def expression_path(self, valobj):
        stream = lldb.SBStream()
        self.assertTrue(valobj.IsValid(), "invalid value object")
        self.assertTrue(valobj.GetExpressionPath(stream))
        return stream.GetData()

    @swiftTest
    @skipEmbeddedSwiftOnWindows
    def test_enum_expression_path(self):
        """The projected case of a payload enum must not drop the enum member
        from its expression path."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        # `frame variable --flat` prints the expression path of each leaf.
        self.expect(
            "frame variable --flat --depth 6",
            substrs=[
                "self.outer.box = boxed",
                "self.outer.box.boxed.leaf.a = 1",
                "self.outer.box.boxed.leaf.b = 2",
            ],
        )

        # The same paths must come out of the SB API, since that is what IDEs
        # feed back into `watchpoint set variable`.
        self_var = self.frame().FindVariable("self")
        outer = self_var.GetChildMemberWithName("outer")
        box = outer.GetChildMemberWithName("box")
        boxed = box.GetChildAtIndex(0)
        leaf = boxed.GetChildMemberWithName("leaf")
        a = leaf.GetChildMemberWithName("a")

        self.assertEqual(self.expression_path(outer), "self.outer")
        self.assertEqual(self.expression_path(box), "self.outer.box")
        self.assertEqual(self.expression_path(boxed), "self.outer.box.boxed")
        self.assertEqual(self.expression_path(leaf), "self.outer.box.boxed.leaf")
        self.assertEqual(self.expression_path(a), "self.outer.box.boxed.leaf.a")

        # A path produced by GetExpressionPath has to round-trip through the
        # expression path parser.
        self.assertEqual(
            self.frame()
            .GetValueForVariablePath("self.outer.box.boxed.leaf.a")
            .GetValueAsSigned(),
            1,
        )

        # ...and work as a `frame variable` argument, which is what a user
        # copying it out of the UI would type.
        self.expect(
            "frame variable self.outer.box.boxed.leaf.a",
            substrs=["self.outer.box.boxed.leaf.a = 1"],
        )

    @swiftTest
    @skipEmbeddedSwiftOnWindows
    @add_test_categories(["watchpoint"])
    def test_watchpoint_on_expression_path(self):
        """A path produced by GetExpressionPath has to be usable as a
        watchpoint location."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        a = (
            self.frame()
            .FindVariable("self")
            .GetChildMemberWithName("outer")
            .GetChildMemberWithName("box")
            .GetChildAtIndex(0)
            .GetChildMemberWithName("leaf")
            .GetChildMemberWithName("a")
        )
        path = self.expression_path(a)
        self.assertEqual(path, "self.outer.box.boxed.leaf.a")

        self.expect(
            "watchpoint set variable " + path,
            substrs=["Watchpoint created", "size = 8"],
        )

        process = self.process()
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)
        self.assertStopReason(
            process.GetSelectedThread().GetStopReason(), lldb.eStopReasonWatchpoint
        )
        self.assertEqual(self.target().GetWatchpointAtIndex(0).GetHitCount(), 1)
