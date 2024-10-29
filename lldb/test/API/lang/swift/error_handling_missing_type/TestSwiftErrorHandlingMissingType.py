import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftErrorHandlingMissingTypes(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'true'))
    def test(self):
        """Test that errors are surfaced"""
        self.build()
        self.expect('settings set symbols.use-swift-clangimporter false')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"))
        frame = thread.frames[0]
        var_object = frame.FindVariable("object", lldb.eNoDynamicValues)
        val = var_object.GetChildAtIndex(1)
        # FIXME: Should be True, for now it's just a string
        self.assertFalse(val.GetError().Fail())
        self.expect('frame var object',
                    substrs=['missing debug info for Clang type', 'FromC'])
        self.expect('frame var enm',
                    substrs=['missing debug info for Clang type', 'ImportedEnum'])
