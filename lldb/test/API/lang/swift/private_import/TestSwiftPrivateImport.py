import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftPrivateImport(TestBase):
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

        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['Library', 'Invisible'])

        # Test that importing the expression context (i.e., the module
        # "a") pulls in the explicit dependencies, but not their
        # private imports.  This comes before the other checks,
        # because type reconstruction will still trigger an import of
        # the "Invisible" module that we can't prevent later one.
        self.expect("expression 1+1")
        self.filecheck('platform shell cat "%s"' % log, __file__)
#       CHECK:  Module import remark: loaded module 'Library'
#       CHECK-NOT: Module import remark: loaded module 'Invisible'

        self.expect("fr var -d run -- x", substrs=["(Invisible.InvisibleStruct)"])
        self.expect("fr var -d run -- y", substrs=["(Library.Conforming)",
                                                   "conforming"])
