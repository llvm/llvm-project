import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    def test_objc_self(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// check self", lldb.SBFileSpec("main.m")
        )
        self.expect("frame variable _ivar", startstr="(int) _ivar = 30")

    @skipUnlessDarwin
    def test_objc_self_capture_idiom(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// check idiomatic self", lldb.SBFileSpec("main.m")
        )
        self.expect("frame variable weakSelf", startstr="(Classic *) weakSelf = 0x")
        self.expect("frame variable self", startstr="(Classic *) self = 0x")
        self.expect("frame variable _ivar", startstr="(int) _ivar = 30")
