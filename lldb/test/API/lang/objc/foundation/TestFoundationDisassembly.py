"""
Test the lldb disassemble command on foundation framework.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FoundationDisassembleTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfAsan
    def test_simple_disasm(self):
        """Test the lldb 'disassemble' command"""
        self.build()

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # Stop at +[NSString stringWithFormat:].
        symbol_name = "+[NSString stringWithFormat:]"
        break_results = lldbutil.run_break_set_command(
            self, "_regexp-break %s" % (symbol_name)
        )

        lldbutil.check_breakpoint_result(
            self, break_results, symbol_name=symbol_name, num_locations=1
        )

        # Stop at -[MyString initWithNSString:].
        lldbutil.run_break_set_by_symbol(
            self,
            "-[MyString initWithNSString:]",
            num_expected_locations=1,
            sym_exact=True,
        )

        # Stop at the "description" selector.
        lldbutil.run_break_set_by_selector(
            self, "description", num_expected_locations=1, module_name="a.out"
        )

        # Stop at -[NSAutoreleasePool release].
        break_results = lldbutil.run_break_set_command(
            self, "_regexp-break -[NSAutoreleasePool release]"
        )
        lldbutil.check_breakpoint_result(
            self,
            break_results,
            symbol_name="-[NSAutoreleasePool release]",
            num_locations=1,
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # First stop is +[NSString stringWithFormat:].
        self.expect(
            "thread backtrace",
            "Stop at +[NSString stringWithFormat:]",
            substrs=["Foundation`+[NSString stringWithFormat:]"],
        )

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")

        self.runCmd("process continue")
        # Skip another breakpoint for +[NSString stringWithFormat:].
        self.runCmd("process continue")

        # Followed by a.out`-[MyString initWithNSString:].
        self.expect(
            "thread backtrace",
            "Stop at a.out`-[MyString initWithNSString:]",
            substrs=["a.out`-[MyString initWithNSString:]"],
        )

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")

        self.runCmd("process continue")

        # Followed by -[MyString description].
        self.expect(
            "thread backtrace",
            "Stop at -[MyString description]",
            substrs=["a.out`-[MyString description]"],
        )

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")

        self.runCmd("process continue")
        # Skip another breakpoint for -[MyString description].
        self.runCmd("process continue")

        # Followed by -[NSAutoreleasePool release].
        self.expect(
            "thread backtrace",
            "Stop at -[NSAutoreleasePool release]",
            substrs=["Foundation`-[NSAutoreleasePool release]"],
        )

        # Do the disassemble for the currently stopped function.
        self.runCmd("disassemble -f")
