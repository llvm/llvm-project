import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftPrivateImport(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test_private_import(self):
        """Test a library with a private import for which there is no debug info"""
        invisible_swift = self.getBuildArtifact("Invisible.swift")
        import shutil
        shutil.copyfile("InvisibleSource.swift", invisible_swift)
        self.build()
        os.unlink(invisible_swift)
        os.unlink(self.getBuildArtifact("Invisible.swiftmodule"))
        os.unlink(self.getBuildArtifact("Invisible.swiftinterface"))

        if lldb.remote_platform:
            wd = lldb.remote_platform.GetWorkingDirectory()
            filename = 'libInvisible.dylib'
            err = lldb.remote_platform.Put(
                lldb.SBFileSpec(self.getBuildArtifact(filename)),
                lldb.SBFileSpec(os.path.join(wd, filename)))
            self.assertFalse(err.Fail(), 'Failed to copy ' + filename)

        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['Library'])
        self.expect("fr var -d run -- x", substrs=["(Invisible.InvisibleStruct)"])
        # FIXME: This crashes LLDB with a Swift DESERIALIZATION FAILURE.
        # self.expect("fr var -d run -- y", substrs=["(Any)"])
