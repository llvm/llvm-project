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
        frame = thread.frame[0]

        symbol = frame.symbol
        # On Windows the function does not have an associated symbol.
        if symbol.IsValid():
            self.assertFalse(symbol.mangled)
            self.assertEqual(symbol.GetDisplayName(), "_Dfunction")

        function = frame.function
        self.assertFalse(function.mangled)
        self.assertEqual(function.GetDisplayName(), "_Dfunction")
