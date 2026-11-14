"""
Tests imported namespaces in C++.
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppNsImport(TestBase):
    @expectedFailureAll(debug_info=["pdb"], bugnumber="llvm.org/pr149498")
    def test_with_run_command(self):
        """Tests imported namespaces in C++."""
        self.build()

        # Get main source file
        src_file = os.path.join(self.getSourceDir(), "main.cpp")
        src_file_spec = lldb.SBFileSpec(src_file)
        self.assertTrue(src_file_spec.IsValid(), "Main source file")

        # Break on main function
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break 0", src_file_spec
        )
        break_1 = target.BreakpointCreateBySourceRegex("// break 1", src_file_spec)
        self.assertTrue(
            break_1.IsValid() and break_1.GetNumLocations() >= 1, VALID_BREAKPOINT
        )

        # Get current fream of the thread at the breakpoint
        frame = thread.GetSelectedFrame()

        # Test imported namespaces
        test_result = frame.EvaluateExpression("n")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 1, "n = 1"
        )

        test_result = frame.EvaluateExpression("N::n")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 1, "N::n = 1"
        )

        test_result = frame.EvaluateExpression("nested")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 3, "nested = 3"
        )

        test_result = frame.EvaluateExpression("anon")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 2, "anon = 2"
        )

        test_result = frame.EvaluateExpression("global")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 4, "global = 4"
        )

        test_result = frame.EvaluateExpression("fun_var")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 9, "fun_var = 9"
        )

        test_result = frame.EvaluateExpression("Fun::fun_var")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 0,
            "Fun::fun_var = 0",
        )

        test_result = frame.EvaluateExpression("not_imported")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 35,
            "not_imported = 35",
        )

        # Currently there is no way to distinguish between "::imported" and "imported" in ClangExpressionDeclMap so this fails
        # test_result = frame.EvaluateExpression("::imported")
        # self.assertTrue(test_result.IsValid() and test_result.GetValueAsSigned() == 89, "::imported = 89")

        test_result = frame.EvaluateExpression("Imported::imported")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 99,
            "Imported::imported = 99",
        )

        test_result = frame.EvaluateExpression("imported")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 99,
            "imported = 99",
        )

        test_result = frame.EvaluateExpression("single")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 3, "single = 3"
        )

        # Continue to second breakpoint
        process.Continue()

        # Get the thread of the process
        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)

        # Get current fream of the thread at the breakpoint
        frame = thread.GetSelectedFrame()

        # Test function inside namespace
        test_result = frame.EvaluateExpression("fun_var")
        self.assertTrue(
            test_result.IsValid() and test_result.GetValueAsSigned() == 5, "fun_var = 5"
        )
