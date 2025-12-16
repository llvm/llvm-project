"""
Test the IR interpreter
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class IRInterpreterTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def time_expression(self, expr, options):
        start = time.time()
        res = self.target.EvaluateExpression(expr, options)
        return res, time.time() - start

    def test_interpreter_timeout(self):
        """Test the timeout logic in the IRInterpreter."""
        self.build()
        self.target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(self.target, VALID_TARGET)

        # A non-trivial infinite loop.
        inf_loop = "for (unsigned i = 0; i < 100; ++i) --i; 1"
        timeout_error = "Reached timeout while interpreting expression"
        options = lldb.SBExpressionOptions()

        # This is an IRInterpreter specific test, so disable the JIT.
        options.SetAllowJIT(False)

        # We use a 500ms timeout.
        options.SetTimeoutInMicroSeconds(500000)
        res, duration_sec = self.time_expression(inf_loop, options)
        self.assertIn(timeout_error, str(res.GetError()))

        # Depending on the machine load the expression might take quite some
        # time, so give the time a generous upper bound.
        self.assertLess(duration_sec, 15)

        # Try a simple one second timeout.
        options.SetTimeoutInMicroSeconds(1000000)
        res, duration_sec = self.time_expression(inf_loop, options)
        self.assertIn(timeout_error, str(res.GetError()))
        # Anything within 5% of 1s is fine, to account for machine differences.
        self.assertGreaterEqual(duration_sec, 0.95)
        self.assertLess(duration_sec, 30)

    def test_interpreter_interrupt(self):
        """Test interrupting the IRInterpreter."""
        self.build()
        self.target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(self.target, VALID_TARGET)

        # A non-trivial infinite loop.
        inf_loop = "for (unsigned i = 0; i < 100; ++i) --i; 1"

        options = lldb.SBExpressionOptions()

        # This is an IRInterpreter specific test, so disable the JIT.
        options.SetAllowJIT(False)

        # Make sure we have a pretty long (10s) timeout so we have a chance to
        # interrupt the interpreted expression.
        options.SetTimeoutInMicroSeconds(10000000)

        self.dbg.RequestInterrupt()

        self.dbg.SetAsync(True)
        res = self.target.EvaluateExpression(inf_loop, options)
        self.dbg.SetAsync(False)

        # Be sure to turn this off again:
        def cleanup():
            if self.dbg.InterruptRequested():
                self.dbg.CancelInterruptRequest()

        self.addTearDownHook(cleanup)

        interrupt_error = "Interrupted while interpreting expression"
        self.assertIn(interrupt_error, str(res.GetError()))

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number("main.c", "// Set breakpoint here")

        # Disable confirmation prompt to avoid infinite wait
        self.runCmd("settings set auto-confirm true")
        self.addTearDownHook(lambda: self.runCmd("settings clear auto-confirm"))

    def build_and_run(self):
        """Test the IR interpreter"""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=False
        )

        self.runCmd("run", RUN_SUCCEEDED)

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="llvm.org/pr21765",
    )
    @add_test_categories(["pyapi"])
    def test_ir_interpreter(self):
        self.build_and_run()

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)

        set_up_expressions = [
            "int $i = 9",
            "int $j = 3",
            "int $k = 5",
            "unsigned long long $ull = -1",
            "unsigned $u = -1",
        ]

        expressions = [
            "$i + $j",
            "$i - $j",
            "$i * $j",
            "$i / $j",
            "$i % $k",
            "$i << $j",
            "$i & $j",
            "$i | $j",
            "$i ^ $j",
            "($ull & -1) == $u",
        ]

        for expression in set_up_expressions:
            self.frame().EvaluateExpression(expression, options)

        func_call = "(int)getpid()"
        if lldbplatformutil.getPlatform() == "windows":
            func_call = "(int)GetCurrentProcessId()"

        for expression in expressions:
            interp_expression = expression
            jit_expression = func_call + "; " + expression

        for expression in expressions:
            interp_expression = expression
            jit_expression = "(int)getpid(); " + expression

            interp_result = (
                self.frame()
                .EvaluateExpression(interp_expression, options)
                .GetValueAsSigned()
            )
            jit_result = (
                self.frame()
                .EvaluateExpression(jit_expression, options)
                .GetValueAsSigned()
            )

            self.assertEqual(
                interp_result, jit_result, "While evaluating " + expression
            )

    def test_type_conversions(self):
        target = self.dbg.GetDummyTarget()
        short_val = target.EvaluateExpression("(short)-1")
        self.assertEqual(short_val.GetValueAsSigned(), -1)
        long_val = target.EvaluateExpression("(long) " + short_val.GetName())
        self.assertEqual(long_val.GetValueAsSigned(), -1)
