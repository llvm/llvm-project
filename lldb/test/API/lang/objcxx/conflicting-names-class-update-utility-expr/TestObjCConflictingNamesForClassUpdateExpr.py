import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    def test(self):
        """
        Tests that running the utility expression that retrieves the Objective-C
        class list works even when user-code contains functions with apparently
        conflicting identifiers (e.g. 'free') but that are not in the global
        scope.

        This is *not* supposed to test what happens when there are actual
        conflicts such as when a user somehow defined their own '::free'
        function.
        """

        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.mm"))

        # First check our side effect variable is in its initial state.
        self.expect_expr("called_function", result_summary='"none"')

        # Get the (dynamic) type of our 'id' variable so that our Objective-C
        # runtime information is updated.
        str_val = self.expect_expr("str")
        dyn_val = str_val.GetDynamicValue(lldb.eDynamicCanRunTarget)
        dyn_type = dyn_val.GetTypeName()

        # Check our side effect variable which should still be in its initial
        # state if none of our trap functions were called.
        # If this is failing, then LLDB called one of the trap functions.
        self.expect_expr("called_function", result_summary='"none"')

        # Double check that our dynamic type is correct. This is done last
        # as the assert message from above is the more descriptive one (it
        # contains the unintentionally called function).
        self.assertEqual(dyn_type, "__NSCFConstantString *")
