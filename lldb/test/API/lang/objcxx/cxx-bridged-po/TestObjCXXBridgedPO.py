import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestObjCXXBridgedPO(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_bridged_type_po(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.mm'))
        self.expect('po num',
                    "did not get the Objective-C object description",
                    substrs=['CFNumber', '0x', '42'])
        pointer_val = str(self.frame().FindVariable('num').GetValue())
        self.expect('po '+pointer_val,
                    "did not get the Objective-C object description",
                    substrs=['CFNumber', '0x', '42'])
