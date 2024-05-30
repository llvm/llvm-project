import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):
    def test_functions_having_dlang_mangling_prefix(self):
        """
        Ensure C functions with a '_D' prefix alone are not mistakenly treated
        as a Dlang mangled name. A proper Dlang mangling will have digits
        immediately following the '_D' prefix.
        """
        self.build()
        _, _, thread, _ = lldbutil.run_to_name_breakpoint(self, "_Dfunction")
        symbol = thread.frame[0].symbol
        self.assertEqual(symbol.GetDisplayName(), "_Dfunction")
