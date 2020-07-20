import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestBitfieldIvars(TestBase):

    mydir = TestBase.compute_mydir(__file__)


    @skipUnlessDarwin
    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.m"))

        self.expect_expr("chb->hb->field1", result_type="unsigned int", result_value="0")

        ## FIXME field2 should have a value of 1
        self.expect("expr chb->hb->field2", matching=False, substrs = ["= 1"]) # this must happen second

        self.expect_expr("hb2->field1", result_type="unsigned int", result_value="10")
        self.expect_expr("hb2->field2", result_type="unsigned int", result_value="3")
        self.expect_expr("hb2->field3", result_type="unsigned int", result_value="4")

        self.expect("frame var *hb2", substrs = [ 'x =', '100',
                                             'field1 =', '10',
                                             'field2 =', '3',
                                             'field3 =', '4'])

    @expectedFailureAll()
    def testExprWholeObject(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.m"))

        ## FIXME expression with individual bit-fields obtains correct values but not with the whole object
        self.expect("expr *hb2", substrs = [ 'x =', '100',
                                             'field1 =', '10',
                                             'field2 =', '3',
                                             'field3 =', '4'])
