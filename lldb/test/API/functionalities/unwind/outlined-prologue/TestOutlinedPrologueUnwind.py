import re
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestOutlinedPrologueUnwind(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(archs=no_match(["arm64", "arm64e", "aarch64"]))
    def test_outlined_prologue_saves_reach_the_unwind_plan(self):
        """Test that register saves done in a prologue are correctly handled by the assembly-scan unwind plan"""
        self.build()
        target, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "callee")

        if not any(
            symbol.GetName() and "OUTLINED_FUNCTION" in symbol.GetName()
            for module in target.module_iter()
            for symbol in module
        ):
            self.skipTest(
                "no outlined function in the binary; the compiler no longer "
                "outlines prologues here, so there is nothing to cover"
            )

        plan = self.res_of("image show-unwind -n middle")
        self.assertIn("Assembly language inspection UnwindPlan:", plan)
        # Sections of the output are separated by blank lines.
        section = plan.split("Assembly language inspection UnwindPlan:", 1)[1]
        section = section.split("\n\n", 1)[0]
        saved = set(re.findall(r"(x\d+)=\[CFA", section))
        self.assertTrue(
            {"x19", "x20", "x21", "x22"}.issubset(saved),
            "the plan for middle() is missing register saves performed by its "
            "outlined prologue; it recovered only %s\n%s" % (sorted(saved), section),
        )

        # middle() overwrites x19-x22 with its own arguments after the outlined
        # prologue spills them.
        caller = thread.GetFrameAtIndex(2)
        self.assertTrue(caller.IsValid(), "the unwind did not reach caller()")
        self.assertIn("caller", caller.GetFunctionName())
        recovered = {
            reg: caller.FindRegister(reg).GetValueAsUnsigned()
            for reg in ("x19", "x20", "x21", "x22")
        }
        self.assertEqual(
            sorted(recovered.values()),
            [0x1111, 0x2222, 0x3333, 0x4444],
            "caller's arguments were read out of registers that middle()'s "
            "outlined prologue spilled and that were never restored: %s" % recovered,
        )

    def res_of(self, command):
        result = lldb.SBCommandReturnObject()
        self.dbg.GetCommandInterpreter().HandleCommand(command, result)
        self.assertTrue(result.Succeeded(), result.GetError())
        return result.GetOutput()
