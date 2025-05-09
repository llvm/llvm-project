import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTaggedPointerCommand(TestBase):
    @no_debug_info_test
    def test(self):
        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.m")
        )

        n1 = thread.GetSelectedFrame().FindVariable("n1")
        self.expect(
            f"lang objc tagged-pointer info {n1.addr}",
            substrs=[
                f"{n1.addr} is tagged",
                "payload = 0x0000000000000012",
                "value = 0x0000000000000001",
                "info bits = 0x0000000000000002",
                "class = __NSCFNumber",
            ],
        )

        self.expect(
            "lang objc tagged-pointer info bogus",
            error=True,
            patterns=["could not convert 'bogus' to a valid address"],
        )

        self.expect(
            "lang objc tagged-pointer info 0x0",
            error=True,
            patterns=["could not convert '0x0' to a valid address"],
        )
