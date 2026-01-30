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

    def test_fpconv(self):
        self.build_and_run()

        interp_options = lldb.SBExpressionOptions()
        interp_options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        interp_options.SetAllowJIT(False)

        jit_options = lldb.SBExpressionOptions()
        jit_options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        jit_options.SetAllowJIT(True)

        set_up_expressions = [
            "int $i = 3",
            "int $n = -3",
            "unsigned $u = 5",
            "long $l = -7",
            "float $f = 9.0625",
            "double $d = 13.75",
            "float $nf = -11.25",
        ]

        expressions = [
            "$i + $f",  # sitofp i32 to float
            "$d - $n",  # sitofp i32 to double
            "$u + $f",  # uitofp i32 to float
            "$u + $d",  # uitofp i32 to double
            "(int)$d",  # fptosi double to i32
            "(int)$f",  # fptosi float to i32
            "(long)$d",  # fptosi double to i64
            "(short)$f",  # fptosi float to i16
            "(long)$nf",  # fptosi float to i64
            "(unsigned short)$f",  # fptoui float to i16
            "(unsigned)$d",  # fptoui double to i32
            "(unsigned long)$d",  # fptoui double to i64
            "(float)$d",  # fptrunc double to float
            "(double)$f",  # fpext float to double
            "(double)$nf",  # fpext float to double
        ]

        for expression in set_up_expressions:
            self.frame().EvaluateExpression(expression, interp_options)

        func_call = "(int)getpid()"
        if lldbplatformutil.getPlatform() == "windows":
            func_call = "(int)GetCurrentProcessId()"

        for expression in expressions:
            interp_expression = expression
            # Calling a function forces the expression to be executed with JIT.
            jit_expression = func_call + "; " + expression

            interp_result = self.frame().EvaluateExpression(
                interp_expression, interp_options
            )
            jit_result = self.frame().EvaluateExpression(jit_expression, jit_options)

            self.assertEqual(
                interp_result.GetValue(),
                jit_result.GetValue(),
                "Values match for " + expression,
            )
            self.assertEqual(
                interp_result.GetTypeName(),
                jit_result.GetTypeName(),
                "Types match for " + expression,
            )

    def test_fpconv_ub(self):
        target = self.dbg.GetDummyTarget()

        set_up_expressions = [
            "float $f = 3e9",
            "double $d = 1e20",
            "float $nf = -1.5",
        ]

        expressions = [
            ("(int)$f", "Conversion error: (float) 3.0E+9 cannot be converted to i32"),
            (
                "(long)$d",
                "Conversion error: (float) 1.0E+20 cannot be converted to i64",
            ),
            (
                "(unsigned)$nf",
                "Conversion error: (float) -1.5 cannot be converted to unsigned i32",
            ),
        ]

        for expression in set_up_expressions:
            target.EvaluateExpression(expression)

        # The IR Interpreter returns an error if a value cannot be converted.
        for expression in expressions:
            result = target.EvaluateExpression(expression[0])
            self.assertIn(expression[1], str(result.GetError()))

        # The conversion should succeed if the destination type can represent the result.
        self.expect_expr(
            "(unsigned)$f", result_type="unsigned int", result_value="3000000000"
        )
        self.expect_expr("(int)$nf", result_type="int", result_value="-1")
