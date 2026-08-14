"""
Test that the expression path of an Optional's payload does not contain the
`some` level that the synthetic children provider hides.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftOptionalExpressionPath(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def expression_path(self, valobj):
        stream = lldb.SBStream()
        self.assertTrue(valobj.IsValid(), "invalid value object")
        self.assertTrue(valobj.GetExpressionPath(stream))
        return stream.GetData()

    def setUp(self):
        TestBase.setUp(self)
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

    @swiftTest
    @skipEmbeddedSwiftOnWindows
    def test_optional_expression_path(self):
        # `frame variable --flat` prints the expression path of each leaf it
        # descends into. There should be no `.some`
        self.expect(
            "frame variable --flat",
            substrs=["holder.opt = ", "holder.n = 5", "scalar = 42"],
        )
        self.expect("frame variable --flat", matching=False, substrs=["some"])

        plain = self.frame().FindVariable("plain")
        self.assertEqual(self.expression_path(plain), "plain")
        self.assertEqual(self.expression_path(plain.GetChildAtIndex(0)), "plain.a")
        self.assertEqual(self.expression_path(plain.GetChildAtIndex(1)), "plain.b")

        opt = self.frame().FindVariable("holder").GetChildMemberWithName("opt")
        self.assertEqual(self.expression_path(opt), "holder.opt")
        self.assertEqual(self.expression_path(opt.GetChildAtIndex(0)), "holder.opt.a")

        # A doubly-wrapped Optional is flattened all the way down, so neither
        # `some` level appears in the path.
        nested = self.frame().FindVariable("nested")
        self.assertEqual(self.expression_path(nested), "nested")
        self.assertEqual(self.expression_path(nested.GetChildAtIndex(0)), "nested.a")
        self.assertEqual(nested.GetChildAtIndex(0).GetValueAsSigned(), 3)

        # A path produced by GetExpressionPath has to resolve back to the same
        # value, both through the SBAPI and as a `frame variable` argument.
        for name in ["plain", "holder"]:
            var = self.frame().FindVariable(name)
            for i in range(var.GetNumChildren()):
                child = var.GetChildAtIndex(i)
                if child.GetNumChildren():
                    continue  # only compare leaves by value
                path = self.expression_path(child)

                resolved = self.frame().GetValueForVariablePath(path)
                self.assertTrue(resolved.IsValid(), "path %s does not resolve" % path)
                self.assertEqual(
                    resolved.GetValueAsSigned(),
                    child.GetValueAsSigned(),
                    "path %s resolves to a different value" % path,
                )

                # The path also has to work in the `frame variable` command,
                # which is what a user copying it out of the UI would type.
                self.expect(
                    "frame variable " + path,
                    substrs=["%s = %s" % (path, child.GetValue())],
                )

        # Hiding `some` from the path must not disturb the value presentation.
        self.expect(
            "frame variable plain scalar empty",
            substrs=["(a = 1, b = 2)", "42", "nil"],
        )
        self.assertEqual(plain.GetNumChildren(), 2)
        self.assertEqual(self.frame().FindVariable("empty").GetNumChildren(), 0)

    @swiftTest
    @skipEmbeddedSwiftOnWindows
    @expectedFailureAll(
        bugnumber="The expression path parser cannot unwrap two Optional levels"
    )
    def test_doubly_nested_optional_round_trips(self):
        """The flattened path of a doubly-wrapped Optional does not yet parse
        back: `nested.a` is rejected on Optional<Optional<Leaf>>."""
        path = self.expression_path(
            self.frame().FindVariable("nested").GetChildAtIndex(0)
        )
        self.assertEqual(path, "nested.a")
        self.assertTrue(self.frame().GetValueForVariablePath(path).IsValid())

    @swiftTest
    @skipEmbeddedSwiftOnWindows
    @add_test_categories(["watchpoint"])
    def test_watchpoint_on_expression_path(self):
        """A path produced by GetExpressionPath has to be usable as a
        watchpoint location."""
        path = self.expression_path(
            self.frame().FindVariable("plain").GetChildAtIndex(0)
        )
        self.assertEqual(path, "plain.a")
        self.expect(
            "watchpoint set variable " + path,
            substrs=["Watchpoint created", "size = 8"],
        )
