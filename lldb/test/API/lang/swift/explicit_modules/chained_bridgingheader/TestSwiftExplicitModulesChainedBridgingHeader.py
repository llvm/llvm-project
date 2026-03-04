import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModulesChainedBridgingHeader(lldbtest.TestBase):
    @swiftTest
    @skipUnlessDarwin # uses a framework, doesn't link with gold
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    def test(self):
        """Test explicit Swift modules with chained bridging headers"""
        self.build()
        # Delete this file to ensure we're not parsing the headers.
        secret = self.getBuildArtifact("secret.h")
        import os
        os.unlink(secret)

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types -f "%s"' % log)
        self.expect("frame variable a", substrs=['a = 23'])
        # FIXME: DWARF types in B.pcm (reachable from the bridging pch) are not found.
        self.expect("frame variable b") # FIXME: substrs=['b = 42'])
        self.expect("expression a", substrs=['a = 23'])
        self.expect("expression b") # FIXME: substrs=['b = 42'])
        self.filecheck_log(log, __file__)
        # CHECK: LogConfiguration{{.*}}Bridging Header PCH{{.*}}a-{{.*}}ChainedBridgingHeader{{.*}}.pch
        # CHECK: LogConfiguration{{.*}}Explicit module map{{.*}}
        # CHECK-NOT: secret
        # CHECK: Import 
