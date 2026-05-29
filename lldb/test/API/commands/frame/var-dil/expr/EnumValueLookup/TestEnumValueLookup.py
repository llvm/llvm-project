"""
Test DIL enum value lookups.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestEnumValueLookup(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_literals(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        self.expect_var_path("enum_one", value="kOne", type="UnscopedEnum")
        self.expect_var_path("UnscopedEnum::kOne", value="kOne", type="UnscopedEnum")
        self.expect_var_path(
            "UnscopedEnumInt8::kOne8", value="kOne8", type="UnscopedEnumInt8"
        )
        self.expect_var_path("ScopedEnum::kOneS", value="kOneS", type="ScopedEnum")
        self.expect_var_path(
            "ns::NSUnscopedEnum::kOneNS", value="kOneNS", type="ns::NSUnscopedEnum"
        )

        # Check the underlying type
        frame = thread.GetFrameAtIndex(0)
        kOne8 = frame.GetValueForVariablePath("UnscopedEnumInt8::kOne8")
        underlying_type = kOne8.GetType().GetEnumerationIntegerType().GetName()
        self.assertEqual(underlying_type, "int8_t")
