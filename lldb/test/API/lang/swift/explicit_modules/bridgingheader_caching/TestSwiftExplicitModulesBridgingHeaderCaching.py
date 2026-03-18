import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModules(lldbtest.TestBase):
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'), bugnumber='rdar://157258485')
    @skipUnlessDarwin
    def test_with_deleted_header(self):
        """Test explicit Swift modules with bridging headers"""
        self.build()
        secret = self.getBuildArtifact("secret")
        import shutil
        shutil.rmtree(secret)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))
        self.expect("frame variable s", substrs=['i = 23'])
        self.expect("frame variable m", substrs=['j = 42'])
        self.expect("expression s", substrs=['i = 23'])
        self.expect("expression m", substrs=['j = 42'])

    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'), bugnumber='rdar://157258485')
    @skipUnlessDarwin
    def test(self):
        """Test explicit Swift modules with bridging headers"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))
        self.expect("frame variable s", substrs=['i = 23'])
        self.expect("frame variable m", substrs=['j = 42'])
        self.expect("expression s", substrs=['i = 23'])
        self.expect("expression m", substrs=['j = 42'])

        # Verify the prefix map was applied: the frame's source file should
        # resolve to the real on-disk path, not the virtual /^src path baked
        # into the debug info by -scanner-prefix-map-paths.
        file_spec = thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec()
        self.assertEqual(file_spec.GetDirectory(), self.getSourceDir())
        self.assertEqual(file_spec.GetFilename(), "main.swift")
