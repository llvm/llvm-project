import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftTaggedPointer(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    # This test depends on NSObject, so it is not available on non-Darwin
    # platforms.
    @skipUnlessDarwin
    # This test exposes a bug in DWARFImporterForClangTypes, which
    # doesn't do type completion correctly. (rdar://118337109)
    @skipIf(setting=('symbols.swift-precise-compiler-invocation', 'true'))
    def test(self):
        self.build()
        # On the bots only, Swift typesystem validation fails.
        self.runCmd("settings set symbols.swift-validate-typesystem false")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect('frame variable -d run -- a', substrs=['Int64(3)'])
        self.expect('frame variable -d run -- b', substrs=['Int64(3)'])
         
        self.expect('frame variable -d run -- c', substrs=['"hi"'])
        self.expect('frame variable -d run -- d', substrs=['"hi"'])
         
        self.expect('expr -d run -- a', substrs=['Int64(3)'])
        self.expect('expr -d run -- b', substrs=['Int64(3)'])
         
        self.expect('expr -d run -- c', substrs=['"hi"'])
        self.expect('expr -d run -- d', substrs=['"hi"'])
