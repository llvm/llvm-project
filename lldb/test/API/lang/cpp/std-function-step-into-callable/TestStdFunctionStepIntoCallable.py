"""
Test stepping into std::function
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibCxxFunctionSteppingIntoCallableTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["libc++"])
    def test(self):
        """Test that std::function as defined by libc++ is correctly printed by LLDB"""
        self.build()

        self.main_source = "main.cpp"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.source_foo_line = line_number(
            self.main_source, '// Source foo start line')
        self.source_lambda_f2_line = line_number(
            self.main_source, '// Source lambda used by f2 start line')
        self.source_lambda_f3_line = line_number(
            self.main_source, '// Source lambda used by f3 start line')
        self.source_bar_operator_line = line_number(
            self.main_source, '// Source Bar::operator()() start line')
        self.source_bar_add_num_line = line_number(
            self.main_source, '// Source Bar::add_num start line')
        self.source_main_invoking_f1 = line_number(
            self.main_source, '// Source main invoking f1')

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line.", self.main_source_spec)

        thread.StepInto()
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.source_main_invoking_f1 ) ;
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source) ;

        thread.StepInto()
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.source_foo_line ) ;
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source) ;
        process.Continue()

        thread.StepInto()
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.source_lambda_f2_line ) ;
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source) ;
        process.Continue()

        thread.StepInto()
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.source_lambda_f3_line ) ;
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source) ;
        process.Continue()

        # TODO reenable this case when std::function formatter supports
        # general callable object case.
        #thread.StepInto()
        #self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.source_bar_operator_line ) ;
        #self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source) ;
        #process.Continue()

        thread.StepInto()
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetLine(), self.source_bar_add_num_line ) ;
        self.assertEqual( thread.GetFrameAtIndex(0).GetLineEntry().GetFileSpec().GetFilename(), self.main_source) ;
        process.Continue()
