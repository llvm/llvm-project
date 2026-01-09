import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftExpressionTypeAlias(lldbtest.TestBase):

    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect('expr -d run -- local', substrs=['Pair<Int>'])
        process.Continue()
        # FIXME!
        self.expect('expr -d run -- local', substrs=['(Int, Int)'])
        process.Continue()
        self.expect("frame var associated", substrs=['A'])
        # FIXME!
        self.expect("expr associated", error=True,
                    substrs=['type for typename "$s1a1BV7MyAliasaD" was not found'])
