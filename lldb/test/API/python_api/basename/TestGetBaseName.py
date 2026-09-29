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
        TestBase.setUp(self)
        self.main_source_file = lldb.SBFileSpec("main.cpp")

    @expectedFailureAll(
        oslist=["windows"],
        bugnumber="https://github.com/llvm/llvm-project/issues/156861",
    )
    def test(self):
        """Test SBFunction.GetBaseName() and SBSymbol.GetBaseName()"""
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", self.main_source_file
        )

        frame0 = thread.GetFrameAtIndex(0)

        # Get both function and symbol
        function = frame0.GetFunction()
        symbol = frame0.GetSymbol()

        # Test consistency between function and symbol basename
        function_basename = function.GetBaseName()
        symbol_basename = symbol.GetBaseName()

        self.assertEqual(function_basename, "templateFunc")
        self.assertEqual(symbol_basename, "templateFunc")
