import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    def test_raw_bytes(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        self.runCmd("type summary add -s '${var.ubyte:x-2}${var.sbyte:x-2}!' Bytes")
        self.expect("v bytes", substrs=[" = 3001!"])

    def test_bad_format(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        self.expect(
            "type summary add -s '${var.ubyte:y}!' Bytes",
            error=True,
            substrs=["invalid llvm format"],
        )
