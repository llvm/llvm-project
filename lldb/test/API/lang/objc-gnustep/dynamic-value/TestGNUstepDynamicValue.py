"""
Test the dynamic type of an Objective-C object with the GNUstep runtime.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestGNUstepDynamicValue(TestBase):
    def test_dynamic_value_from_api(self):
        """The dynamic type is read out of the runtime's class structures."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        frame = self.frame()
        static_value = frame.FindVariable("object", lldb.eNoDynamicValues)
        self.assertTrue(static_value.IsValid(), "found the variable")
        self.assertEqual(static_value.GetTypeName(), "Base *")

        dynamic_value = static_value.GetDynamicValue(lldb.eDynamicCanRunTarget)
        self.assertTrue(dynamic_value.IsValid(), "resolved a dynamic value")
        self.assertEqual(dynamic_value.GetTypeName(), "Derived *")

        # A variable whose dynamic and static types agree stays unchanged.
        base = frame.FindVariable("base", lldb.eNoDynamicValues).GetDynamicValue(
            lldb.eDynamicCanRunTarget
        )
        self.assertEqual(base.GetTypeName(), "Base *")

    def test_dynamic_value_from_command(self):
        """`frame variable` reports the same dynamic type as the API."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        self.expect(
            "frame variable -d run-target object", substrs=["(Derived *) object"]
        )
        self.expect(
            "frame variable -d no-dynamic-values object",
            substrs=["(Base *) object"],
        )
