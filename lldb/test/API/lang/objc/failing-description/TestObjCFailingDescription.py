import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))

        log = self.getBuildArtifact("expr.log")
        self.runCmd(f"log enable lldb expr -f {log}")

        self.expect(
            "expr -O -- bad",
            substrs=["`po` was unsuccessful, running `p` instead\n", "(Bad *) 0x"],
        )
        self.filecheck(
            f"platform shell cat {log}", __file__, f"-check-prefix=CHECK-EXPR"
        )
        # CHECK-EXPR: Object description fallback due to error: could not evaluate print object function: expression interrupted

        self.expect(
            "dwim-print -O -- bad",
            substrs=["`po` was unsuccessful, running `p` instead\n", "_lookHere = NO"],
        )
        self.filecheck(
            f"platform shell cat {log}", __file__, f"-check-prefix=CHECK-DWIM-PRINT"
        )
        # CHECK-DWIM-PRINT: Object description fallback due to error: could not evaluate print object function: expression interrupted
