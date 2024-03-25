"""
Test lldb Python API SBValue::Cast(SBType) for C++ types.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CppValueCastTestCase(TestBase):
    @skipIf(bugnumber="llvm.org/PR36714")
    @add_test_categories(["pyapi"])
    def test_value_cast_with_virtual_inheritance(self):
        """Test SBValue::Cast(SBType) API for C++ types with virtual inheritance."""
        self.build(dictionary=self.d_virtual)
        self.setTearDownCleanup(dictionary=self.d_virtual)
        self.do_sbvalue_cast(self.exe_name)

    @add_test_categories(["pyapi"])
    def test_value_cast_with_regular_inheritance(self):
        """Test SBValue::Cast(SBType) API for C++ types with regular inheritance."""
        self.build(dictionary=self.d_regular)
        self.setTearDownCleanup(dictionary=self.d_regular)
        self.do_sbvalue_cast(self.exe_name)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        # Find the line number to break for main.c.
        self.source = "sbvalue-cast.cpp"
        self.line = line_number(self.source, "// Set breakpoint here.")
        self.exe_name = self.testMethodName
        self.d_virtual = {
            "CXX_SOURCES": self.source,
            "EXE": self.exe_name,
            "CFLAGS_EXTRAS": "-DDO_VIRTUAL_INHERITANCE",
        }
        self.d_regular = {"CXX_SOURCES": self.source, "EXE": self.exe_name}

    def do_sbvalue_cast(self, exe_name):
        """Test SBValue::Cast(SBType) API for C++ types."""
        exe = self.getBuildArtifact(exe_name)

        # Create a target from the debugger.

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        breakpoint = target.BreakpointCreateByLocation(self.source, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Now launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        # Find DerivedA and DerivedB types.
        typeA = target.FindFirstType("DerivedA")
        typeB = target.FindFirstType("DerivedB")
        self.DebugSBType(typeA)
        self.DebugSBType(typeB)
        self.assertTrue(typeA)
        self.assertTrue(typeB)
        error = lldb.SBError()

        # First stop is for DerivedA instance.
        threads = lldbutil.get_threads_stopped_at_breakpoint(process, breakpoint)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        frame0 = thread.GetFrameAtIndex(0)

        tellerA = frame0.FindVariable("teller", lldb.eNoDynamicValues)
        self.DebugSBValue(tellerA)
        self.assertEqual(
            tellerA.GetChildMemberWithName("m_base_val").GetValueAsUnsigned(error, 0),
            20,
        )

        if self.TraceOn():
            for child in tellerA:
                print("child name:", child.GetName())
                print(child)

        # Call SBValue.Cast() to obtain instanceA.
        instanceA = tellerA.Cast(typeA.GetPointerType())
        self.DebugSBValue(instanceA)

        # Iterate through all the children and print their values.
        if self.TraceOn():
            for child in instanceA:
                print("child name:", child.GetName())
                print(child)
        a_member_val = instanceA.GetChildMemberWithName("m_a_val")
        self.DebugSBValue(a_member_val)
        self.assertEqual(a_member_val.GetValueAsUnsigned(error, 0), 10)

        # Second stop is for DerivedB instance.
        threads = lldbutil.continue_to_breakpoint(process, breakpoint)
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        frame0 = thread.GetFrameAtIndex(0)

        tellerB = frame0.FindVariable("teller", lldb.eNoDynamicValues)
        self.DebugSBValue(tellerB)
        self.assertEqual(
            tellerB.GetChildMemberWithName("m_base_val").GetValueAsUnsigned(error, 0),
            12,
        )

        if self.TraceOn():
            for child in tellerB:
                print("child name:", child.GetName())
                print(child)

        # Call SBValue.Cast() to obtain instanceB.
        instanceB = tellerB.Cast(typeB.GetPointerType())
        self.DebugSBValue(instanceB)

        # Iterate through all the children and print their values.
        if self.TraceOn():
            for child in instanceB:
                print("child name:", child.GetName())
                print(child)
        b_member_val = instanceB.GetChildMemberWithName("m_b_val")
        self.DebugSBValue(b_member_val)
        self.assertEqual(b_member_val.GetValueAsUnsigned(error, 0), 36)
