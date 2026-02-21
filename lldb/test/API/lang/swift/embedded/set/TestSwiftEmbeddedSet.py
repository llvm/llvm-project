import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedSet(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test Set data formatter in embedded Swift."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        s = frame.FindVariable("s")
        lldbutil.check_variable(self, s, False, summary="5 values")

        self.assertEqual(s.GetNumChildren(), 5)
        elements = set()
        for i in range(5):
            child = s.GetChildAtIndex(i)
            elements.add(int(child.GetValue()))
        self.assertEqual(elements, {1, 2, 3, 4, 5})
        
        emptySet = frame.FindVariable("emptySet")
        lldbutil.check_variable(self, emptySet, False, summary="0 values")
