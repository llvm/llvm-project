"""
Test that lldb persistent types works correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class PersistenttypesTestCase(TestBase):

    mydir = os.path.join("expression_command", "persistent_types")

    def test_persistent_types(self):
        """Test that lldb persistent types works correctly."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name main")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expression struct $foo { int a; int b; };")

        self.expect("expression struct $foo $my_foo; $my_foo.a = 2; $my_foo.b = 3;",
                    startstr = "(int) $0 = 3")

        self.expect("expression $my_foo",
                    substrs = ['a = 2', 'b = 3'])

        self.runCmd("expression typedef int $bar")

        self.expect("expression $bar i = 5; i",
                    startstr = "($bar) $1 = 5")

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
