"""
Test dwim-print with objc instances.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @skipUnlessDarwin
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect("dwim-print parent", substrs=["_child = 0x"])
        self.expect(
            "dwim-print parent.child", patterns=[r'_name = 0x[0-9a-f]+ @"Seven"']
        )

    @skipUnlessDarwin
    def test_with_summary(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.runCmd("type summary add -s 'Parent of ${var._child._name}' 'Parent *'")
        self.expect("dwim-print parent", matching=False, substrs=["_child = 0x"])
        self.expect("dwim-print parent", substrs=['Parent of @"Seven"'])
