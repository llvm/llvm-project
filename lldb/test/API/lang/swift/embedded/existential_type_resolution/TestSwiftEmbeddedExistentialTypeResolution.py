import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedExistentialTypeResolution(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect('frame variable pStruct', substrs=['a.S', 'structField = 111'])
        self.expect('frame variable pClass', substrs=['a.C', 'classField = 222'])
        self.expect('frame variable pSubclass', substrs=['a.Sub', 'a.C', 'classField = 222', 'subField = 333'])
        self.expect('frame variable pEnumFirst', substrs=['a.E', 'first'])
        self.expect('frame variable pEnumSecond', substrs=['a.E', 'second'])
