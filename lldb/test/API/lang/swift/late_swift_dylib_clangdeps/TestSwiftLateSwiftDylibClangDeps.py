from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftLateSwiftDylibClangDeps(TestBase):
    @skipIf(macos_version=["<", "14.0"])
    @skipUnlessDarwin
    @swiftTest
    @skipIfDarwinEmbedded 
    def test(self):
        """Test that the reflection metadata cache is invalidated
        when new DWARF debug info is available"""
        self.build()
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("loader.swift"))

        # Initialize SwiftASTContext before loading the dylib.
        self.runCmd("setting set symbols.swift-enable-ast-context false")
        self.expect("v fromClang",
                    substrs=["missing debug info", "FromClang"])

        bkpt = target.BreakpointCreateByLocation(
            lldb.SBFileSpec('dylib.swift'), 5)
        threads = lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect("v fromClang", substrs=['42'])
        self.expect("frame select 1")
        # Note that in earlier versions the lookup was a global one and
        # thus re-evaluating the variable after adding dylib would
        # have produced a different result. Currently these lookups
        # are per-module so this expectedly still fails.
        self.expect("v fromClang",
                    substrs=["missing debug info", "FromClang"])
