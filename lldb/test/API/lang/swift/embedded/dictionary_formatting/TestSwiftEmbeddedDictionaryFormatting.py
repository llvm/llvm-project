import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedDictionaryFormatting(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test Dictionary data formatter children in embedded Swift."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        dictionary = frame.FindVariable("dict")
        lldbutil.check_variable(
            self, dictionary, False, summary="3 key/value pairs"
        )

        self.assertEqual(dictionary.GetNumChildren(), 3)
        keys = set()
        values = set()
        for i in range(3):
            child = dictionary.GetChildAtIndex(i)
            key = child.GetChildMemberWithName("key")
            value = child.GetChildMemberWithName("value")
            keys.add(int(key.GetValue()))
            values.add(int(value.GetValue()))
        self.assertEqual(keys, {1, 2, 3})
        self.assertEqual(values, {10, 20, 30})

        # rdar://170883616
        # emptyDict = frame.FindVariable("emptyDict")
        # lldbutil.check_variable(
        #     self, emptyDict, False, summary="0 key/value pairs"
        # )
