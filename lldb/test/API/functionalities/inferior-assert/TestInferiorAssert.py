"""Test that lldb functions correctly after the inferior has asserted."""

import lldb
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbplatformutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class AssertingInferiorTestCase(TestBase):

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21793: need to implement support for detecting assertion / abort on Windows")
    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm"],
        bugnumber="llvm.org/pr25338")
    @expectedFailureAll(bugnumber="llvm.org/pr26592", triple='^mips')
    def test_inferior_asserting(self):
        """Test that lldb reliably catches the inferior asserting (command)."""
        self.build()
        self.inferior_asserting()

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21793: need to implement support for detecting assertion / abort on Windows")
    @expectedFailureAndroid(
        api_levels=list(
            range(
                16 +
                1)))  # b.android.com/179836
    def test_inferior_asserting_register(self):
        """Test that lldb reliably reads registers from the inferior after asserting (command)."""
        self.build()
        self.inferior_asserting_registers()

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21793: need to implement support for detecting assertion / abort on Windows")
    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm"],
        triple=no_match(".*-android"),
        bugnumber="llvm.org/pr25338")
    @expectedFailureAll(bugnumber="llvm.org/pr26592", triple='^mips')
    def test_inferior_asserting_disassemble(self):
        """Test that lldb reliably disassembles frames after asserting (command)."""
        self.build()
        self.inferior_asserting_disassemble()

    @add_test_categories(['pyapi'])
    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21793: need to implement support for detecting assertion / abort on Windows")
    def test_inferior_asserting_python(self):
        """Test that lldb reliably catches the inferior asserting (Python API)."""
        self.build()
        self.inferior_asserting_python()

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21793: need to implement support for detecting assertion / abort on Windows")
    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm"],
        triple=no_match(".*-android"),
        bugnumber="llvm.org/pr25338")
    @expectedFailureAll(bugnumber="llvm.org/pr26592", triple='^mips')
    def test_inferior_asserting_expr(self):
        """Test that the lldb expression interpreter can read from the inferior after asserting (command)."""
        self.build()
        self.inferior_asserting_expr()

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21793: need to implement support for detecting assertion / abort on Windows")
    @expectedFailureAll(
        oslist=["linux"],
        archs=["arm"],
        triple=no_match(".*-android"),
        bugnumber="llvm.org/pr25338")
    @expectedFailureAll(bugnumber="llvm.org/pr26592", triple='^mips')
    def test_inferior_asserting_step(self):
        """Test that lldb functions correctly after stepping through a call to assert()."""
        self.build()
        self.inferior_asserting_step()

    def set_breakpoint(self, line):
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line, num_expected_locations=1, loc_exact=True)

    def check_stop_reason(self):
        matched = lldbplatformutil.match_android_device(
            self.getArchitecture(), valid_api_levels=list(range(1, 16 + 1)))
        if matched:
            # On android until API-16 the abort() call ended in a sigsegv
            # instead of in a sigabrt
            stop_reason = 'signal SIGSEGV'
        else:
            stop_reason = 'signal SIGABRT'

        target = self.dbg.GetTargetAtIndex(0)
        process = target.GetProcess()
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)

        pattern = "stop reason = (" + stop_reason + "|hit program assert)"

        # The stop reason of the thread should be an abort signal or exception.
        self.expect("thread list", STOPPED_DUE_TO_ASSERT,
                    patterns=[pattern],
                    substrs=['stopped'])

        return pattern

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number of the call to assert.
        self.line = line_number('main.c', '// Assert here.')

    def inferior_asserting(self):
        """Inferior asserts upon launching; lldb should catch the event and stop."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)
        stop_reason = self.check_stop_reason()

        # And it should report a backtrace that includes the assert site.
        self.expect("thread backtrace all",
                    patterns=[stop_reason],
                    substrs=['main', 'argc', 'argv'])

        # And it should report the correct line number.
        self.expect("thread backtrace all",
                    patterns=[stop_reason],
                    substrs=['main.c:%d' % self.line])

    def inferior_asserting_python(self):
        """Inferior asserts upon launching; lldb should catch the event and stop."""
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now launch the process, and do not stop at entry point.
        # Both argv and envp are null.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonSignal)
        if not thread:
            self.fail("Fail to stop the thread upon assert")

        if self.TraceOn():
            lldbutil.print_stacktrace(thread)

    def inferior_asserting_registers(self):
        """Test that lldb can read registers after asserting."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd("run", RUN_SUCCEEDED)
        self.check_stop_reason()

        # change current frame to frame 0, since recognizer might have selected
        # different frame.
        self.runCmd("frame select 0", RUN_SUCCEEDED)

        # lldb should be able to read from registers from the inferior after
        # asserting.
        lldbplatformutil.check_first_register_readable(self)

    def inferior_asserting_disassemble(self):
        """Test that lldb can disassemble frames after asserting."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        target.LaunchSimple(None, None, self.get_process_working_directory())
        self.check_stop_reason()

        process = target.GetProcess()
        self.assertTrue(process.IsValid(), "current process is valid")

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        lastframeID = thread.GetFrameAtIndex(
            thread.GetNumFrames() - 1).GetFrameID()

        isi386Arch = False
        if "i386" in self.getArchitecture():
            isi386Arch = True

        # lldb should be able to disassemble frames from the inferior after
        # asserting.
        for frame in thread:
            self.assertTrue(frame.IsValid(), "current frame is valid")

            self.runCmd("frame select " +
                        str(frame.GetFrameID()), RUN_SUCCEEDED)

            # Don't expect the function name to be in the disassembly as the assert
            # function might be a no-return function where the PC is past the end
            # of the function and in the next function. We also can't back the PC up
            # because we don't know how much to back it up by on targets with opcodes
            # that have differing sizes
            pc_backup_offset = 1
            if frame.GetFrameID() == 0:
                pc_backup_offset = 0
            if isi386Arch:
                if lastframeID == frame.GetFrameID():
                    pc_backup_offset = 0
            self.expect(
                "disassemble -a %s" %
                (frame.GetPC() -
                 pc_backup_offset),
                substrs=['<+0>: '])

    def check_expr_in_main(self, thread):
        depth = thread.GetNumFrames()
        for i in range(depth):
            frame = thread.GetFrameAtIndex(i)
            self.assertTrue(frame.IsValid(), "current frame is valid")
            if self.TraceOn():
                print(
                    "Checking if function %s is main" %
                    frame.GetFunctionName())

            if 'main' == frame.GetFunctionName():
                frame_id = frame.GetFrameID()
                self.runCmd("frame select " + str(frame_id), RUN_SUCCEEDED)
                self.expect("expression argc", substrs=['(int)', ' = 1'])
                self.expect("expression hello_world", substrs=['Hello'])
                self.expect("expression argv[0]", substrs=['a.out'])
                self.expect("expression null_ptr", substrs=['= 0x0'])
                return True
        return False

    def inferior_asserting_expr(self):
        """Test that the lldb expression interpreter can read symbols after asserting."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        target.LaunchSimple(None, None, self.get_process_working_directory())
        self.check_stop_reason()

        process = target.GetProcess()
        self.assertTrue(process.IsValid(), "current process is valid")

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        # The lldb expression interpreter should be able to read from addresses
        # of the inferior after a call to assert().
        self.assertTrue(
            self.check_expr_in_main(thread),
            "cannot find 'main' in the backtrace")

    def inferior_asserting_step(self):
        """Test that lldb functions correctly after stepping through a call to assert()."""
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        self.set_breakpoint(self.line)
        target.LaunchSimple(None, None, self.get_process_working_directory())

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['main.c:%d' % self.line,
                             'stop reason = breakpoint'])

        self.runCmd("next")
        stop_reason = self.check_stop_reason()

        # lldb should be able to read from registers from the inferior after
        # asserting.
        if "x86_64" in self.getArchitecture():
            self.expect("register read rbp", substrs=['rbp = 0x'])
        if "i386" in self.getArchitecture():
            self.expect("register read ebp", substrs=['ebp = 0x'])

        process = target.GetProcess()
        self.assertTrue(process.IsValid(), "current process is valid")

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        # The lldb expression interpreter should be able to read from addresses
        # of the inferior after a call to assert().
        self.assertTrue(
            self.check_expr_in_main(thread),
            "cannot find 'main' in the backtrace")

        # And it should report the correct line number.
        self.expect("thread backtrace all",
                    patterns=[stop_reason],
                    substrs=['main.c:%d' % self.line])
