"""
Test that parray of a struct with an embedded char array works.
This was failing because the "live address" of the child elements
was calculated incorrectly - as a offset from the pointer live 
address.  It only happened for char[] children because they used
GetAddressOf which relies on the live address.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestParrayVrsCharArrayChild(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_parray_struct_with_char_array_child(self):
        """This is the basic test for does parray get the char values right."""
        self.build()
        self.main_source_file = lldb.SBFileSpec("main.c")
        self.do_array_test()

    def do_array_test(self):
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "Set a breakpoint here", self.main_source_file)

        frame = thread.GetFrameAtIndex(0)
 
        self.expect("expr -Z 3 -- struct_ptr",
                    substrs = ['before = 112, var = "abcd", after = 221',
                               'before = 313, var = "efgh", after = 414',
                               'before = 515, var = "ijkl", after = 616'])

        
