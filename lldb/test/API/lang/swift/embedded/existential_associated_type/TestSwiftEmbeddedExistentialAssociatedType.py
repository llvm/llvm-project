import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedExistentialAssociatedType(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("next") # rdar://170158298
        self.expect('frame variable a', substrs=['[a.Q]', '2 values', 's = (i = 1)'])
        self.expect('frame variable x0', substrs=['a.SmallContainer', 's = (i = 1)'])
        self.expect('frame variable x', substrs=['a.Small', 'i = 1'])
