import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftResiliencePrivateMethod(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        # FXIME!
        self.expect("expression priv", error=True,
                    substrs=["Couldn't look up symbols",
                             "dispatch thunk of a.Resilient.priv.getter"])
