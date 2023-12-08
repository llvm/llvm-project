import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *


class TestCase(TestBase):
    def test_invalid_arg(self):
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "thread select -1", error=True, startstr="error: Invalid thread index '-1'"
        )
        self.expect(
            "thread select 0x1ffffffff",
            error=True,
            startstr="error: Invalid thread index '0x1ffffffff'",
        )
        # Parses but not a valid thread id.
        self.expect(
            "thread select 0xffffffff",
            error=True,
            startstr="error: invalid thread #0xffffffff.",
        )
