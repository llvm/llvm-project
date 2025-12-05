import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftObservation(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    def test(self):
        """Test that types with private discriminators read from the file cache work"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        # FIXME: Private discriminators are UUIDs in DWARF and pointers
        # in Reflection metafdata, making tham not comparable.
        # rdar://74374120
        self.expect("settings set symbols.swift-enable-ast-context false")
        self.expect(
            "settings set target.experimental.swift-read-metadata-from-file-cache true"
        )
        r = self.frame().FindVariable("r")
        extent = r.GetChildAtIndex(0)
        self.assertEqual(extent.GetName(), "extent")
        context = extent.GetChildAtIndex(0)
        self.assertEqual(context.GetName(), "context")
