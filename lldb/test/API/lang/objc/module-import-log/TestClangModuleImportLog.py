import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    def test(self):
        self.build()

        log = self.getBuildArtifact("types.log")
        self.runCmd(f"log enable lldb types -f {log}")

        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))

        self.runCmd("expression -lobjc -- @import Foundation")
        self.filecheck_log(log, __file__)
        # CHECK: Importing Clang module Foundation from {{.+}}/Foundation-{{[^/]+}}.pcm
