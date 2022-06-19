import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppIncompleteTypes(TestBase):

    @skipIf(compiler="gcc")
    def test_limit_debug_info(self):
        self.build()
        frame = self.get_test_frame('limit')

        value_f = frame.EvaluateExpression("f")
        self.assertTrue(
            value_f.IsValid(),
            "'expr f' results in a valid SBValue object")
        self.assertSuccess(value_f.GetError(), "'expr f' is successful")

        value_a = frame.EvaluateExpression("a")
        self.assertTrue(
            value_a.IsValid(),
            "'expr a' results in a valid SBValue object")
        self.assertSuccess(value_a.GetError(), "'expr a' is successful")

    @skipIf(compiler="gcc")
    # Clang on Windows asserts in external record layout in this case.
    @skipIfWindows
    def test_partial_limit_debug_info(self):
        self.build()
        frame = self.get_test_frame('nolimit')

        value_f = frame.EvaluateExpression("f")
        self.assertTrue(
            value_f.IsValid(),
            "'expr f' results in a valid SBValue object")
        self.assertSuccess(value_f.GetError(), "'expr f' is successful")

        value_a = frame.EvaluateExpression("a")
        self.assertTrue(
            value_a.IsValid(),
            "'expr a' results in a valid SBValue object")
        self.assertSuccess(value_a.GetError(), "'expr a' is successful")

    def get_test_frame(self, exe):
        # Get main source file
        src_file = "main.cpp"
        src_file_spec = lldb.SBFileSpec(src_file)

        (target, process, thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
                "break here", src_file_spec, exe_name = exe)
        # Get frame for current thread
        return thread.GetSelectedFrame()
