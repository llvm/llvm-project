import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @swiftTest
    @skipUnlessDarwin
    def test_objc_self(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "check self", lldb.SBFileSpec("main.swift"))
        self.expect("frame variable _prop", startstr="(Int) _prop = 30")

    @swiftTest
    @skipUnlessDarwin
    def test_objc_self_capture_idiom(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "check idiomatic self", lldb.SBFileSpec("main.swift"))
        self.expect("frame variable self", startstr="(a.Classic) self = 0x")
        self.expect("frame variable _prop", startstr="(Int) _prop = 30")
