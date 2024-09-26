import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftClangImporterProtocolComposition(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """Test that protocol composition types can be resolved
           through the Swift language runtime"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))
        obj = self.frame().FindVariable("obj", lldb.eDynamicDontRunTarget)
        p = obj.GetChildAtIndex(0, lldb.eDynamicDontRunTarget, False)
        some = p.GetChildAtIndex(0, lldb.eDynamicDontRunTarget, False)
        synth_obj = p.GetChildMemberWithName('object', lldb.eDynamicDontRunTarget)
        i = synth_obj.GetChildMemberWithName('i', lldb.eDynamicDontRunTarget)
        lldbutil.check_variable(self, i, value="23")
