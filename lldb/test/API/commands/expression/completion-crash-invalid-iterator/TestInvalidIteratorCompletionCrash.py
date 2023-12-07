from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipIf  # rdar://problem/53931074
    def test(self):
        self.build()
        target = self.createTestTarget()
        callee_break = target.BreakpointCreateByName(
            "SomeClass::SomeClass(ParamClass)", None
        )
        self.assertTrue(callee_break.GetNumLocations() > 0)
        self.runCmd("run", RUN_SUCCEEDED)

        to_complete = "e ParamClass"
        self.dbg.GetCommandInterpreter().HandleCompletion(
            to_complete, len(to_complete), 0, -1, lldb.SBStringList()
        )
