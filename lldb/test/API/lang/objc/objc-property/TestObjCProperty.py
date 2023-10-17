"""
Use lldb Python API to verify that expression evaluation for property references uses the correct getters and setters
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCPropertyTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Find the line number to break for main.c.
        self.source_name = "main.m"

    @add_test_categories(["pyapi"])
    def test_objc_properties(self):
        """Test that expr uses the correct property getters and setters"""
        if self.getArchitecture() == "i386":
            self.skipTest("requires modern objc runtime")

        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        main_bkpt = target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here.", lldb.SBFileSpec(self.source_name)
        )
        self.assertTrue(
            main_bkpt and main_bkpt.GetNumLocations() == 1, VALID_BREAKPOINT
        )

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        threads = lldbutil.get_threads_stopped_at_breakpoint(process, main_bkpt)
        self.assertEquals(len(threads), 1)
        thread = threads[0]
        frame = thread.GetFrameAtIndex(0)

        mine = frame.FindVariable("mine")
        self.assertTrue(mine.IsValid())
        access_count = mine.GetChildMemberWithName("_access_count")
        self.assertTrue(access_count.IsValid())
        start_access_count = access_count.GetValueAsUnsigned(123456)
        self.assertNotEqual(start_access_count, 123456)

        #
        # The first set of tests test calling the getter & setter of
        # a property that actually only has a getter & setter and no
        # @property.
        #
        nonexistant_value = frame.EvaluateExpression("mine.nonexistantInt", False)
        nonexistant_error = nonexistant_value.GetError()
        self.assertSuccess(nonexistant_error)
        nonexistant_int = nonexistant_value.GetValueAsUnsigned(123456)
        self.assertEquals(nonexistant_int, 6)

        # Calling the getter function would up the access count, so make sure
        # that happened.

        new_access_count = access_count.GetValueAsUnsigned(123456)
        self.assertEquals(new_access_count - start_access_count, 1)
        start_access_count = new_access_count

        #
        # Now call the setter, then make sure that
        nonexistant_change = frame.EvaluateExpression("mine.nonexistantInt = 10", False)
        nonexistant_error = nonexistant_change.GetError()
        self.assertSuccess(nonexistant_error)

        # Calling the setter function would up the access count, so make sure
        # that happened.

        new_access_count = access_count.GetValueAsUnsigned(123456)
        self.assertEquals(new_access_count - start_access_count, 1)
        start_access_count = new_access_count

        #
        # Now we call the getter of a property that is backed by an ivar,
        # make sure it works and that we actually update the backing ivar.
        #

        backed_value = frame.EvaluateExpression("mine.backedInt", False)
        backed_error = backed_value.GetError()
        self.assertSuccess(backed_error)
        backing_value = mine.GetChildMemberWithName("_backedInt")
        self.assertTrue(backing_value.IsValid())
        self.assertEqual(
            backed_value.GetValueAsUnsigned(12345),
            backing_value.GetValueAsUnsigned(23456),
        )

        value_from_typedef = frame.EvaluateExpression("typedefd.backedInt", False)
        self.assertSuccess(value_from_typedef.GetError())
        self.assertEqual(
            value_from_typedef.GetValueAsUnsigned(12345),
            backing_value.GetValueAsUnsigned(23456),
        )

        unbacked_value = frame.EvaluateExpression("mine.unbackedInt", False)
        unbacked_error = unbacked_value.GetError()
        self.assertSuccess(unbacked_error)

        idWithProtocol_value = frame.EvaluateExpression("mine.idWithProtocol", False)
        idWithProtocol_error = idWithProtocol_value.GetError()
        self.assertSuccess(idWithProtocol_error)
        self.assertEquals(idWithProtocol_value.GetTypeName(), "id")

        # Make sure that class property getter works as expected
        value = frame.EvaluateExpression("BaseClass.classInt", False)
        self.assertSuccess(value.GetError())
        self.assertEquals(value.GetValueAsUnsigned(11111), 123)

        # Make sure that class property setter works as expected
        value = frame.EvaluateExpression("BaseClass.classInt = 234", False)
        self.assertSuccess(value.GetError())

        # Verify that setter above actually worked
        value = frame.EvaluateExpression("BaseClass.classInt", False)
        self.assertSuccess(value.GetError())
        self.assertEquals(value.GetValueAsUnsigned(11111), 234)

        # Test that accessing two distinct class and instance properties that
        # share the same name works.
        self.expect_expr("mine.propConflict", result_value="4")
        self.expect_expr("BaseClass.propConflict", result_value="6")
