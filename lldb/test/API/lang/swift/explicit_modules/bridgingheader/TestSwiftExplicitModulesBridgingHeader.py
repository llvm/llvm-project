import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModules(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'), bugnumber='rdar://157258485')
    def test_with_deleted_header(self):
        """Test explicit Swift modules with bridging headers"""
        self.build()
        secret = self.getBuildArtifact("secret")
        import shutil
        shutil.rmtree(secret)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.runCmd('settings set symbols.swift-validate-typesystem false')
        self.expect("frame variable s", substrs=['i = 23'])
        self.expect("frame variable m", substrs=['j = 42'])
        # FIXME: Can this be avoided?
        self.expect("expression s", error=True,
                    substrs=["could not find file", "referenced by AST file", ".pch"])
        self.expect("expression m", error=True,
                    substrs=["could not find file", "referenced by AST file", ".pch"])
        self.filecheck('platform shell cat "%s"' % log, __file__)
        # CHECK: LogConfiguration
        # CHECK-NOT: secret
        # CHECK: Import 

    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'), bugnumber='rdar://157258485')
    def test(self):
        """Test explicit Swift modules with bridging headers"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))
        self.runCmd('settings set symbols.swift-validate-typesystem false')
        self.expect("frame variable s", substrs=['i = 23'])
        self.expect("frame variable m", substrs=['j = 42'])
        self.expect("expression s", substrs=['i = 23'])
        self.expect("expression m", substrs=['j = 42'])
