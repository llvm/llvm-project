import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_invalid_arg(self):
        self.build()

        lldbutil.run_to_source_breakpoint(self, '// break here', lldb.SBFileSpec("main.cpp"))
        self.expect("process signal az", error=True, startstr="error: Invalid signal argument 'az'.")
        self.expect("process signal 0x1ffffffff", error=True, startstr="error: Invalid signal argument '0x1ffffffff'.")
        self.expect("process signal 0xffffffff", error=True, startstr="error: Invalid signal argument '0xffffffff'.")
