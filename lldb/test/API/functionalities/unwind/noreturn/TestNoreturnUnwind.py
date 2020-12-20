"""
Test that we can backtrace correctly with 'noreturn' functions on the stack
"""

from __future__ import print_function


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NoreturnUnwind(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    @skipIfWindows  # clang-cl does not support gcc style attributes.
    # clang does not preserve LR in noreturn functions, making unwinding impossible
    @skipIf(compiler="clang", archs=['arm'], oslist=['linux'])
    @expectedFailureAll(bugnumber="llvm.org/pr33452", triple='^mips')
    def test(self):
        """Test that we can backtrace correctly with 'noreturn' functions on the stack"""
        self.build()
        self.setTearDownCleanup()

        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if not process:
            self.fail("SBTarget.Launch() failed")

        if process.GetState() != lldb.eStateStopped:
            self.fail("Process should be in the 'stopped' state, "
                      "instead the actual state is: '%s'" %
                      lldbutil.state_type_to_str(process.GetState()))

        thread = process.GetThreadAtIndex(0)
        abort_frame_number = 0
        for f in thread.frames:
            # Some C libraries mangle the abort symbol into __GI_abort.
            if f.GetFunctionName() in ["abort", "__GI_abort"]:
                break
            abort_frame_number = abort_frame_number + 1

        if self.TraceOn():
            print("Backtrace once we're stopped:")
            for f in thread.frames:
                print("  %d %s" % (f.GetFrameID(), f.GetFunctionName()))

        # I'm going to assume that abort() ends up calling/invoking another
        # function before halting the process.  In which case if abort_frame_number
        # equals 0, we didn't find abort() in the backtrace.
        if abort_frame_number == len(thread.frames):
            self.fail("Unable to find abort() in backtrace.")

        func_c_frame_number = abort_frame_number + 1
        if thread.GetFrameAtIndex(
                func_c_frame_number).GetFunctionName() != "func_c":
            self.fail("Did not find func_c() above abort().")

        # This depends on whether we see the func_b inlined function in the backtrace
        # or not.  I'm not interested in testing that aspect of the backtrace here
        # right now.

        if thread.GetFrameAtIndex(
                func_c_frame_number +
                1).GetFunctionName() == "func_b":
            func_a_frame_number = func_c_frame_number + 2
        else:
            func_a_frame_number = func_c_frame_number + 1

        if thread.GetFrameAtIndex(
                func_a_frame_number).GetFunctionName() != "func_a":
            self.fail("Did not find func_a() above func_c().")

        main_frame_number = func_a_frame_number + 1

        if thread.GetFrameAtIndex(
                main_frame_number).GetFunctionName() != "main":
            self.fail("Did not find main() above func_a().")
