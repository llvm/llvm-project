import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprFromNonZeroFrame(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # Requires DWARF debug information.
    @skipIfWindows
    def test(self):
        """
        Tests that we can use SBFrame::EvaluateExpression on a frame
        that we're not stopped in, even if thread-plans run as part of
        parsing the expression (e.g., when running static initializers).
        """
        self.build()

        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "return 5", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(1)

        # Using a function pointer inside the expression ensures we
        # emit a ptrauth static initializer on arm64e into the JITted
        # expression. The thread-plan that runs for this static
        # initializer should save/restore the current execution context
        # frame (which in this test is frame #1).
        result = frame.EvaluateExpression("int (*fptr)() = &func; fptr()")
        self.assertTrue(result.GetError().Success())
        self.assertEqual(result.GetValueAsSigned(), 5)
