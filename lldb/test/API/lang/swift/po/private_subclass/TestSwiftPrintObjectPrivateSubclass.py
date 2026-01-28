import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @swiftTest
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        log = self.getBuildArtifact("expr.log")
        self.runCmd(f"log enable lldb expr -f {log}")

        self.expect("frame variable -O x", substrs=["Easy as pie"])
        self.filecheck(f"platform shell cat {log}", __file__)

        # Clear the log.
        self.runCmd(f"log disable lldb expr")
        os.unlink(log)
        self.runCmd(f"log enable lldb expr -f {log}")

        self.expect("dwim-print -O -- x", substrs=["Easy as pie"])
        self.filecheck(f"platform shell cat {log}", __file__)

        # Verify po used the mangled name of the static type - which is public,
        # and not the private dynamic type.
        #
        # CHECK: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a10PublicBaseCD")
        # CHECK: stringForPrintObject(_:mangledTypeName:) succeeded
