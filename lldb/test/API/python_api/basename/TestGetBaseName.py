"""
Test SBFunction::GetBaseName() and SBSymbol::GetBaseName() APIs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class GetBaseNameTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break on.
        self.line1 = line_number(
            "main.cpp", "// Find the line number for breakpoint 1 here."
        )

    def test(self):
        """Test SBFunction.GetBaseName() and SBSymbol.GetBaseName()"""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Create a breakpoint inside the C++ namespaced function.
        breakpoint1 = target.BreakpointCreateByLocation("main.cpp", self.line1)

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, self.get_process_working_directory())

        # Get stopped thread and frame
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Get both function and symbol
        function = frame0.GetFunction()
        symbol = frame0.GetSymbol()

        # Test consistency between function and symbol basename
        function_basename = function.GetBaseName()
        symbol_basename = symbol.GetBaseName()

        self.assertEqual(function_basename, "templateFunc")
        self.assertEqual(symbol_basename, "templateFunc")

        self.trace("Function basename:", function_basename)
        self.trace("Symbol basename:", symbol_basename)
