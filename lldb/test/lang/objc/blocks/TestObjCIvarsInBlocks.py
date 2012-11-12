"""Test printing ivars and ObjC objects captured in blocks that are made in methods of an ObjC class."""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class TestObjCIvarsInBlocks(TestBase):

    mydir = os.path.join("lang", "objc", "blocks")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    # This test requires the 2.0 runtime, so it will fail on i386.
    @expectedFailurei386
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test printing the ivars of the self when captured in blocks"""
        self.buildDsym()
        self.ivars_in_blocks()

    @python_api_test
    # This test requires the 2.0 runtime, so it will fail on i386.
    @expectedFailurei386
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test printing the ivars of the self when captured in blocks"""
        self.buildDwarf()
        self.ivars_in_blocks()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line numbers to break inside main().
        self.main_source = "main.m"
        self.class_source = "ivars-in-blocks.m"
        self.class_source_file_spec = lldb.SBFileSpec(self.class_source)

    def ivars_in_blocks (self):
        """Test printing the ivars of the self when captured in blocks"""
        exe = os.path.join(os.getcwd(), "a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateBySourceRegex ('// Break here inside the block.', self.class_source_file_spec)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        process = target.LaunchSimple (None, None, os.getcwd())
        self.assertTrue (process, "Created a process.")
        self.assertTrue (process.GetState() == lldb.eStateStopped, "Stopped it too.")

        thread_list = lldbutil.get_threads_stopped_at_breakpoint (process, breakpoint)
        self.assertTrue (len(thread_list) == 1)
        thread = thread_list[0]
        
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue (frame, "frame 0 is valid")
        
        # First use the FindVariable API to see if we can find the ivar by undecorated name:
        direct_blocky = frame.GetValueForVariablePath ("blocky_ivar")
        self.assertTrue(direct_blocky, "Found direct access to blocky_ivar.")
        
        # Now get it as a member of "self" and make sure the two values are equal:
        self_var = frame.GetValueForVariablePath ("self")
        self.assertTrue (self_var, "Found self in block.")
        indirect_blocky = self_var.GetChildMemberWithName ("blocky_ivar")
        self.assertTrue (indirect_blocky, "Found blocky_ivar through self")
        
        error = lldb.SBError()
        direct_value = direct_blocky.GetValueAsSigned(error)
        self.assertTrue (error.Success(), "Got direct value for blocky_ivar")

        indirect_value = indirect_blocky.GetValueAsSigned (error)
        self.assertTrue (error.Success(), "Got indirect value for blocky_ivar")
        
        self.assertTrue (direct_value == indirect_value, "Direct and indirect values are equal.")

        # Now make sure that we can get at the captured ivar through the expression parser.
        # Doing a little trivial math will force this into the real expression parser:
        direct_expr = frame.EvaluateExpression ("blocky_ivar + 10")
        self.assertTrue (direct_expr, "Got blocky_ivar through the expression parser")
        
        # Again, get the value through self directly and make sure they are the same:
        indirect_expr = frame.EvaluateExpression ("self->blocky_ivar + 10")
        self.assertTrue (indirect_expr, "Got blocky ivar through expression parser using self.")
        
        direct_value = direct_expr.GetValueAsSigned (error)
        self.assertTrue (error.Success(), "Got value from direct use of expression parser")

        indirect_value = indirect_expr.GetValueAsSigned (error)
        self.assertTrue (error.Success(), "Got value from indirect access using the expression parser")

        self.assertTrue (direct_value == indirect_value, "Direct ivar access and indirect through expression parser produce same value.")
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
