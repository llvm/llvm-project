"""Tests IR interpreter handling of basic floating point operations (fadd, fsub, etc)."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class FPEvalTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        self.jit_opts = lldb.SBExpressionOptions()
        self.jit_opts.SetAllowJIT(True)
        self.no_jit_opts = lldb.SBExpressionOptions()
        self.no_jit_opts.SetAllowJIT(False)
        # Find the line number to break inside main().
        self.line = line_number('main.c', '// Set break point at this line.')

    def test(self):
        """Test floating point expressions while jitter is disabled."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break inside the main.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1, loc_exact=True)

        
        value = self.frame().EvaluateExpression("a + b", self.no_jit_opts)

        self.runCmd("run", RUN_SUCCEEDED)
        # test double
        self.expect("expr --allow-jit false  -- a + b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['double', '44'])
        self.expect("expr --allow-jit false  -- a - b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['double', '40'])
        self.expect("expr --allow-jit false  -- a / b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['double', '21'])
        self.expect("expr --allow-jit false  -- a * b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['double', '84'])
        self.expect("expr --allow-jit false  -- a + 2", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['double', '44'])
        self.expect("expr --allow-jit false  -- a > b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['true'])
        self.expect("expr --allow-jit false  -- a >= b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['true'])
        self.expect("expr --allow-jit false  -- a < b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['false'])
        self.expect("expr --allow-jit false  -- a <= b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['false'])
        self.expect("expr --allow-jit false  -- a == b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['false'])
        self.expect("expr --allow-jit false  -- a != b", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['true'])
        
        # test single
        self.expect("expr --allow-jit false  -- f + q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['float', '44'])
        self.expect("expr --allow-jit false  -- f - q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['float', '40'])
        self.expect("expr --allow-jit false  -- f / q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['float', '21'])
        self.expect("expr --allow-jit false  -- f * q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['float', '84'])
        self.expect("expr --allow-jit false  -- f + 2", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['float', '44'])
        self.expect("expr --allow-jit false  -- f > q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['true'])
        self.expect("expr --allow-jit false  -- f >= q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['true'])
        self.expect("expr --allow-jit false  -- f < q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['false'])
        self.expect("expr --allow-jit false  -- f <= q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['false'])
        self.expect("expr --allow-jit false  -- f == q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['false'])
        self.expect("expr --allow-jit false  -- f != q", VARIABLES_DISPLAYED_CORRECTLY,
                    substrs=['true'])

        # compare jit and interpreter output
        self.assertTrue(self.process().IsValid())
        thread = lldbutil.get_stopped_thread(self.process(), lldb.eStopReasonBreakpoint)
        self.assertTrue(thread.IsValid())
        frame = thread.GetSelectedFrame()
        self.assertTrue(frame.IsValid())
       
        dividents = [42, 79, 666]
        divisors = [1.618, 2.718281828, 3.1415926535, 6.62607015]

        for x in dividents:
            for y in divisors:
                vardef = "double x = {0}, y = {1};".format(x, y) 
                v1 = frame.EvaluateExpression("{0}; eval(x, y, 2)".format(vardef), self.jit_opts)
                v2 = frame.EvaluateExpression("{0}; x / y".format(vardef), self.no_jit_opts)
                self.assertTrue(v1.IsValid() and v2.IsValid())
                self.assertTrue(str(v1.GetData()) == str(v2.GetData()))

