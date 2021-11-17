"""Test breakpoint on a class constructor; and variable list the this object."""



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ClassTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_with_run_command(self):
        """Test 'frame variable this' when stopped on a class constructor."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on the ctor function of class C.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The test suite sometimes shows that the process has exited without stopping.
        #
        # CC=clang ./dotest.py -v -t class_types
        # ...
        # Process 76604 exited with status = 0 (0x00000000)
        self.runCmd("process status")

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        # We should be stopped on the ctor function of class C.
        self.expect(
            "frame variable --show-types this",
            VARIABLES_DISPLAYED_CORRECTLY,
            substrs=[
                'C *',
                ' this = '])

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Use Python APIs to create a breakpoint by (filespec, line)."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        filespec = target.GetExecutable()
        self.assertTrue(filespec, VALID_FILESPEC)

        fsDir = os.path.normpath(filespec.GetDirectory())
        fsFile = filespec.GetFilename()

        self.assertTrue(fsDir == os.path.dirname(self.getBuildArtifact())
                        and fsFile == "a.out",
                        "FileSpec matches the executable")

        bpfilespec = lldb.SBFileSpec("main.cpp", False)

        breakpoint = target.BreakpointCreateByLocation(bpfilespec, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Verify the breakpoint just created.
        self.expect(str(breakpoint), BREAKPOINT_CREATED, exe=False,
                    substrs=['main.cpp',
                             str(self.line)])

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.Launch() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        # The stop reason of the thread should be breakpoint.
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        # The filename of frame #0 should be 'main.cpp' and the line number
        # should be 93.
        self.expect("%s:%d" % (lldbutil.get_filenames(thread)[0],
                               lldbutil.get_line_numbers(thread)[0]),
                    "Break correctly at main.cpp:%d" % self.line, exe=False,
                    startstr="main.cpp:")
        # clang compiled code reported main.cpp:94?
        # startstr = "main.cpp:93")

        # We should be stopped on the breakpoint with a hit count of 1.
        self.assertEqual(breakpoint.GetHitCount(), 1, BREAKPOINT_HIT_ONCE)

        process.Continue()

    def test_with_expr_parser(self):
        """Test 'frame variable this' and 'expr this' when stopped inside a constructor."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # rdar://problem/8516141
        # Is this a case of clang (116.1) generating bad debug info?
        #
        # Break on the ctor function of class C.
        # self.expect("breakpoint set -M C", BREAKPOINT_CREATED,
        #    startstr = "Breakpoint created: 1: name = 'C'")

        # Make the test case more robust by using line number to break,
        # instead.
        lldbutil.run_break_set_by_file_and_line(
            self, None, self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        lldbutil.check_breakpoint(self, bpno = 1, expected_hit_count = 1)

        # Continue on inside the ctor() body...
        self.runCmd("register read pc")
        self.runCmd("thread step-over")

        # Verify that 'frame variable this' gets the data type correct.
        self.expect("frame variable this", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['C *'])

        # Verify that frame variable --show-types this->m_c_int behaves
        # correctly.
        self.runCmd("register read pc")
        self.runCmd("expr m_c_int")
        self.expect(
            "frame variable --show-types this->m_c_int",
            VARIABLES_DISPLAYED_CORRECTLY,
            startstr='(int) this->m_c_int = 66')

        # Verify that 'expression this' gets the data type correct.
        self.expect("expression this", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['C *'])

        # rdar://problem/8430916
        # expr this->m_c_int returns an incorrect value
        #
        # Verify that expr this->m_c_int behaves correctly.
        self.expect("expression this->m_c_int", VARIABLES_DISPLAYED_CORRECTLY,
                    patterns=['\(int\) \$[0-9]+ = 66'])

    def test_with_constructor_name(self):
        """Test 'frame variable this' and 'expr this' when stopped inside a constructor."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        filespec = target.GetExecutable()
        self.assertTrue(filespec, VALID_FILESPEC)

        fsDir = os.path.normpath(filespec.GetDirectory())
        fsFile = filespec.GetFilename()

        self.assertTrue(fsDir == os.path.dirname(self.getBuildArtifact())
                        and fsFile == "a.out",
                        "FileSpec matches the executable")

        bpfilespec = lldb.SBFileSpec("main.cpp", False)

        breakpoint = target.BreakpointCreateByLocation(bpfilespec, self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Verify the breakpoint just created.
        self.expect(str(breakpoint), BREAKPOINT_CREATED, exe=False,
                    substrs=['main.cpp',
                             str(self.line)])

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.Launch() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        # The stop reason of the thread should be breakpoint.
        thread = lldbutil.get_stopped_thread(
            process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)

        frame = thread.frames[0]
        self.assertTrue(frame.IsValid(), "Got a valid frame.")

        self.assertIn("C::C", frame.name,
                      "Constructor name includes class name.")
