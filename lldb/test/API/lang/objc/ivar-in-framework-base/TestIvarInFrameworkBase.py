import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestIvarInFrameworkBase(TestBase):
    """
    Tests whether LLDB's data inspection commands can correctly retrieve
    information about ivars from the Objective-C runtime.
    In this test-case we have a base class type for which we don't have access
    to the debug-info of the implementation (mimicking the scenario of subclassing
    a type from a system framework). LLDB won't be able to see the backing ivar for
    'fooProp' from just debug-info, but it will fall back on the runtime to get the
    necessary information.
    """

    def test_frame_var(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect("frame variable *bar", substrs=["_fooProp = 10", "_barProp = 15"])

    def test_expr(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect_expr(
            "*bar",
            result_type="Bar",
            result_children=[
                ValueCheck(
                    name="Foo",
                    children=[
                        ValueCheck(name="NSObject"),
                        ValueCheck(name="_fooProp", value="10"),
                    ],
                ),
                ValueCheck(name="_barProp", value="15"),
            ],
        )
