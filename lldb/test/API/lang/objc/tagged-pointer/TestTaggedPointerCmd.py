import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTaggedPointerCommand(TestBase):

    @no_debug_info_test
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,"// break here", lldb.SBFileSpec("main.m"))

        self.expect("lang objc tagged-pointer info bogus", error=True,
                    patterns=["could not convert 'bogus' to a valid address"])

        self.expect("lang objc tagged-pointer info 0x0", error=True,
                    patterns=["could not convert '0x0' to a valid address"])
