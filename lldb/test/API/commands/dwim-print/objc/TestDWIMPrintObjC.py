"""
Test dwim-print with objc instances.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):
    @requireDarwin
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.expect("dwim-print parent", substrs=["_child = 0x"])
        self.expect(
            "dwim-print parent.child", patterns=[r'_name = 0x[0-9a-f]+ @"Seven"']
        )

    @requireDarwin
    def test_with_summary(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.m"))
        self.runCmd("type summary add -s 'Parent of ${var._child._name}' 'Parent *'")
        self.expect("dwim-print parent", matching=False, substrs=["_child = 0x"])
        self.expect("dwim-print parent", substrs=['Parent of @"Seven"'])

    @requireDarwin
    def test_property_backing_storage(self):
        """A property backed by a differently-named ivar (DW_TAG_property /
        DW_AT_property_forward) should take the frame-variable fast path,
        instead of unconditionally falling back to `expression`. Covers the
        same three synthesis variants as
        clang/test/DebugInfo/ObjC/property-backing-storage.m."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here for backing storage", lldb.SBFileSpec("main.m")
        )
        self.runCmd("settings set dwim-print-verbosity full")

        self.expect(
            "dwim-print obj.declaredBacking",
            substrs=["ran `frame variable obj._customDeclaredIvar`", "(int) 42"],
        )
        self.expect(
            "dwim-print obj.undeclaredBacking",
            substrs=["ran `frame variable obj._customUndeclaredIvar`", "(int) 7"],
        )
        self.expect(
            "dwim-print obj.implicitBacking",
            substrs=["ran `frame variable obj._implicitBacking`", "(int) 99"],
        )
