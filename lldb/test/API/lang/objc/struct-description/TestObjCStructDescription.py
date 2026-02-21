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
            "vo pair",
            substrs=[
                "warning: `po` was unsuccessful, running `p` instead\n",
                "(Pair) pair = (f = 2, e = 3)",
            ],
        )
        self.filecheck(f"platform shell cat {log}", __file__, f"-check-prefix=CHECK-VO")
        # CHECK-VO: Object description fallback due to error: not a pointer type

        self.expect(
            "expr -O -- pair",
            substrs=[
                "warning: `po` was unsuccessful, running `p` instead\n",
                "(Pair)  (f = 2, e = 3)",
            ],
        )
        self.filecheck(
            f"platform shell cat {log}", __file__, f"-check-prefix=CHECK-EXPR"
        )
        # CHECK-EXPR: Object description fallback due to error: not a pointer type
