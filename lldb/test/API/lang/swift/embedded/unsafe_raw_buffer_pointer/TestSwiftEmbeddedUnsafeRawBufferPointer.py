import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedUnsafeRawBufferPointer(TestBase):
    @skipIf(bugnumber="rdar://170883698")
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test UnsafeRawBufferPointer formatter in embedded Swift."""
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        # 3 Ints * 8 bytes = 24 bytes.
        buffer = frame.FindVariable("buffer")
        lldbutil.check_variable(self, buffer, False, summary="24 values")

        # Verify children are accessible.
        self.assertEqual(buffer.GetNumChildren(), 24)
