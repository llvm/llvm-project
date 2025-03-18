import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedClassTypeResolution(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect('frame variable s', substrs=['a.Sub', 'a.Super', 'superField = 42', 'subField = 100'])
        self.expect('frame variable p', substrs=['a.Sub', 'a.Super', 'superField = 42', 'subField = 100'])
