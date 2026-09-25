"""
Test SBSymbolContext APIs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SymbolContextAPITestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to of function 'c'.
        self.line = line_number(
            "main.c", '// Find the line number of function "c" here.'
        )

    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24778")
    def test(self):
        """Exercise SBSymbolContext API extensively."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        # Frame #0 should be on self.line.
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "c", bkpt_module=exe)
        frame0 = thread.GetFrameAtIndex(0)
        self.assertEqual(frame0.GetLineEntry().GetLine(), self.line)

        # Now get the SBSymbolContext from this frame.  We want everything. :-)
        context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)
        self.assertTrue(context)

        # Get the description of this module.
        module = context.GetModule()
        desc = lldbutil.get_description(module)
        self.expect(desc, "The module should match", exe=False, substrs=[exe])

        compileUnit = context.GetCompileUnit()
        self.expect(
            str(compileUnit),
            "The compile unit should match",
            exe=False,
            substrs=[self.getSourcePath("main.c")],
        )

        function = context.GetFunction()
        self.assertTrue(function)

        block = context.GetBlock()
        self.assertTrue(block)

        lineEntry = context.GetLineEntry()
        self.expect(
            lineEntry.GetFileSpec().GetDirectory(),
            "The line entry should have the correct directory",
            exe=False,
            substrs=[self.mydir],
        )
        self.expect(
            lineEntry.GetFileSpec().GetFilename(),
            "The line entry should have the correct filename",
            exe=False,
            substrs=["main.c"],
        )
        self.assertEqual(
            lineEntry.GetLine(), self.line, "The line entry's line number should match "
        )

        symbol = context.GetSymbol()
        self.assertTrue(
            function.GetName() == symbol.GetName() and symbol.GetName() == "c",
            "The symbol name should be 'c'",
        )

        sc_list = lldb.SBSymbolContextList()
        sc_list.Append(context)
        self.assertEqual(len(sc_list), 1)
        for sc in sc_list:
            self.assertEqual(lineEntry, sc.GetLineEntry())
