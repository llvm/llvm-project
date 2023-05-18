import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    # LLDB ends up calling the user-defined function (but at least doesn't
    # crash).
    @skipIf(macos_version=["<", "13.0"])
    def test(self):
        """
        Tests LLDB's behaviour if the user defines their own conflicting
        objc_copyRealizedClassList_nolock function.
        """

        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        # Get the (dynamic) type of our 'id' variable so that our Objective-C
        # runtime information is updated.
        str_val = self.expect_expr("custom_class")
        dyn_val = str_val.GetDynamicValue(lldb.eDynamicCanRunTarget)

        # We should have retrieved the proper class list even in presence of
        # the user-defined function.
        self.assertEqual(dyn_val.GetTypeName(), "CustomClass *")
