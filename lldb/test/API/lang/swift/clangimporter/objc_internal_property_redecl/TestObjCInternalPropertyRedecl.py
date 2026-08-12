import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestObjCInternalPropertyRedecl(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @requireNotEmbeddedSwift
    @skipUnlessDarwin
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    def test(self):
        """Test that the import of a Clang private companion framework
           module works in the expression evaluator"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['TestFramework'])

        self.expect("expression o.privateDetail",
                    substrs=["private detail: internal implementation info"])
