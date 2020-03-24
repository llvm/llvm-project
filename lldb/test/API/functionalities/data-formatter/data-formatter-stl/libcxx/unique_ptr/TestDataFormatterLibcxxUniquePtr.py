"""
Test lldb data formatter for libc++ std::unique_ptr.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LibcxUniquePtrDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @add_test_categories(["libc++"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()

        (self.target, self.process, _, bkpt) = lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))

        self.expect("frame variable up_empty",
            substrs=['(std::unique_ptr<int, std::default_delete<int> >) up_empty = nullptr {',
                               '__value_ = ',
                               '}'])

        self.expect("frame variable up_int",
            substrs=['(std::unique_ptr<int, std::default_delete<int> >) up_int = 10 {',
                               '__value_ = ',
                               '}'])

        self.expect("frame variable up_int_ref",
            substrs=['(std::unique_ptr<int, std::default_delete<int> > &) up_int_ref = 10: {',
                               '__value_ = ',
                               '}'])

        self.expect("frame variable up_int_ref_ref",
            substrs=['(std::unique_ptr<int, std::default_delete<int> > &&) up_int_ref_ref = 10: {',
                               '__value_ = ',
                               '}'])

        self.expect("frame variable up_str",
            substrs=['up_str = "hello" {',
                               '__value_ = ',
                               '}'])
