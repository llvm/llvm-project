"""
Make sure if we have two classes with the same base name the
dynamic value calculator doesn't confuse them
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class DynamicValueSameBaseTestCase(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_same_basename_this(self):
        """Test that the we use the full name to resolve dynamic types."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.cpp")
        self.sample_test()

    def sample_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Break here to get started", self.main_source_file
        )

        # Set breakpoints in the two class methods and run to them:
        namesp_bkpt = target.BreakpointCreateBySourceRegex(
            "namesp function did something.", self.main_source_file
        )
        self.assertEqual(
            namesp_bkpt.GetNumLocations(), 1, "Namespace breakpoint invalid"
        )

        virtual_bkpt = target.BreakpointCreateBySourceRegex(
            "Virtual function did something.", self.main_source_file
        )
        self.assertEqual(
            virtual_bkpt.GetNumLocations(), 1, "Virtual breakpoint invalid"
        )

        threads = lldbutil.continue_to_breakpoint(process, namesp_bkpt)
        self.assertEqual(len(threads), 1, "Didn't stop at namespace breakpoint")

        frame = threads[0].frame[0]
        namesp_this = frame.FindVariable("this", lldb.eDynamicCanRunTarget)
        # Clang specifies the type of this as "T *", gcc as "T * const". This
        # erases the difference.
        namesp_type = namesp_this.GetType().GetUnqualifiedType()
        self.assertEqual(
            namesp_type.GetName(),
            "namesp::Virtual *",
            "Didn't get the right dynamic type",
        )

        threads = lldbutil.continue_to_breakpoint(process, virtual_bkpt)
        self.assertEqual(len(threads), 1, "Didn't stop at virtual breakpoint")

        frame = threads[0].frame[0]
        virtual_this = frame.FindVariable("this", lldb.eDynamicCanRunTarget)
        virtual_type = virtual_this.GetType().GetUnqualifiedType()
        self.assertEqual(
            virtual_type.GetName(), "Virtual *", "Didn't get the right dynamic type"
        )
