"""
Test that we can have persistent pointer variables
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class PersistentPtrUpdateTestCase(TestBase):

    def test(self):
        """Test that we can have persistent pointer variables"""
        self.build()

        def cleanup():
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        self.runCmd('break set -p here')

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("expr void* $foo = 0")

        self.runCmd("continue")

        self.expect("expr $foo", substrs=['$foo', '0x0'])
