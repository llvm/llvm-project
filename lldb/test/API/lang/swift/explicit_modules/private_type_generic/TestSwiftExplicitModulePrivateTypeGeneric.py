import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExplicitModulePrivateTypeGeneric(lldbtest.TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIfWindows
    def test(self):
        """Test frame variable of a generic struct specialized to a
        private type from another explicitly-built module."""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"),
            extra_images=["Dylib"])
        log = self.getBuildArtifact("types.log")
        self.expect('log enable lldb types symbols -f "%s"' % log)
        self.expect("expr -d run -- s", error=True)
        # FIXME: Expression shouldn't depend on all local variables.
        self.expect("expression 1+1", error=True)
        self.filecheck_log(log, __file__)
        # CHECK: Turning off implicit modules
        # CHECK: ReconstructType
        # CHECK: Turning on implicit modules
