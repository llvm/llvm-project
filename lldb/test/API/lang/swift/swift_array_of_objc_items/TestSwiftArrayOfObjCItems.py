import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestCase(TestBase):

    @swiftTest
    @skipEmbeddedSwift
    @skipUnlessDarwin
    def test(self):
        """Test that a Swift array of ObjC items prints correctly."""

        self.build()
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.m")
        )

        frame = thread.selected_frame
        items = frame.var("items").children
        self.assertEqual(len(items), 2)

        def get_name(item):
            return item.member["_name"].summary

        self.assertEqual(get_name(items[0]), '@"hello"')
        self.assertEqual(get_name(items[1]), '@"world"')
