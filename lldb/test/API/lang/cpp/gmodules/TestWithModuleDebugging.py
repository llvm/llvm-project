import lldb
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWithGmodulesDebugInfo(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(bugnumber="llvm.org/pr36146", oslist=["linux"], archs=["i386"])
    @add_test_categories(["gmodules"])
    def test_specialized_typedef_from_pch(self):
        self.build()

        src_file = os.path.join(self.getSourceDir(), "main.cpp")
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "breakpoint file")

        # Get the path of the executable
        exe_path = self.getBuildArtifact("a.out")

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on interesting line
        breakpoint = target.BreakpointCreateBySourceRegex(
            "break here", src_file_spec)
        self.assertTrue(
            breakpoint.IsValid() and breakpoint.GetNumLocations() >= 1,
            VALID_BREAKPOINT)

        # Launch the process
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), PROCESS_IS_VALID)

        # Get the thread of the process
        self.assertState(process.GetState(), lldb.eStateStopped)
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint condition")

        # Get frame for current thread
        frame = thread.frames[0]

        testValue = frame.EvaluateExpression("test")
        self.assertTrue(
            testValue.GetError().Success(),
            "Test expression value invalid: %s" %
            (testValue.GetError().GetCString()))
        self.assertEqual(
            testValue.GetTypeName(), "IntContainer",
            "Test expression type incorrect")

        memberValue = testValue.GetChildMemberWithName("storage")
        self.assertTrue(
            memberValue.GetError().Success(),
            "Member value missing or invalid: %s" %
            (testValue.GetError().GetCString()))
        self.assertEqual(
            memberValue.GetTypeName(), "int",
            "Member type incorrect")
        self.assertEqual(
            42,
            memberValue.GetValueAsSigned(),
            "Member value incorrect")

        testValue = frame.EvaluateExpression("bar")
        self.assertTrue(
            testValue.GetError().Success(),
            "Test expression value invalid: %s" %
            (testValue.GetError().GetCString()))
        self.assertEqual(
            testValue.GetTypeName(), "Foo::Bar",
            "Test expression type incorrect")

        memberValue = testValue.GetChildMemberWithName("i")
        self.assertTrue(
            memberValue.GetError().Success(),
            "Member value missing or invalid: %s" %
            (testValue.GetError().GetCString()))
        self.assertEqual(
            memberValue.GetTypeName(), "int",
            "Member type incorrect")
        self.assertEqual(
            123,
            memberValue.GetValueAsSigned(),
            "Member value incorrect")

