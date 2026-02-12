import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    @skipUnlessDarwin
    def test_synthetic(self):
        self.build()
        if self.TraceOn():
            self.expect("log enable -v lldb formatters")

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        frame = thread.selected_frame
        account = frame.var("acc")
        self.assertEqual(account.num_children, 1)
        self.assertEqual(account.child[0].name, "username")

        self.expect("v acc", matching=False, substrs=["password"])

    @skipUnlessDarwin
    def test_update_reuse(self):
        self.build()

        log = self.getBuildArtifact("formatter.log")
        self.runCmd(f"log enable lldb formatters -f {log}")

        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        frame = thread.selected_frame
        account = frame.var("acc")
        self.assertEqual(account.num_children, 1)

        self.filecheck(f"platform shell cat {log}", __file__)
        # CHECK: Bytecode formatter returned eReuse from `update` (type: `Account`, name: `acc`)
