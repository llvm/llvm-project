import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftFirstExprModuleLoad(lldbtest.TestBase):

    @skipIf(oslist='windows')
    @swiftTest
    @skipUnlessFoundation
    def test_unknown_self_objc_ref(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        # FIXME: This runs into a known bug where the mangled names
        # for private types from reflection metadata contain pointer
        # values as discriminators, but the ones from debug info /
        # Swift modules contain UUIDs. (rdar://74374120)
        self.runCmd("settings set symbols.swift-validate-typesystem false")
        self.expect('expr -d run -- self', substrs=['NSAttributedString'])
