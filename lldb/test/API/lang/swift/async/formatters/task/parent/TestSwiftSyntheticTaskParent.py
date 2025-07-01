import re
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


ADDR_PATTERN = "(0x[0-9a-f]{6,})"

class TestCase(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_name_breakpoint(self, "breakHere")

        # First breakpoint hit occurrs in a root task, with no parent.
        self.expect("task info", substrs=["parent = nil"])
        root_task = self._extract("task info", f"address = {ADDR_PATTERN}")

        # Continue to the next hit of the same breakpoint, which is called from
        # an async let child task.
        process.Continue()
        parent_of_child_task = self._extract("task info", f"parent = {ADDR_PATTERN}")

        # Ensure the parent of the child is the same as the root task.
        self.assertEqual(root_task, parent_of_child_task)

    def _extract(self, command: str, pattern: str) -> str:
        ret = lldb.SBCommandReturnObject()
        self.ci.HandleCommand(command, ret)
        match = re.search(pattern, ret.GetOutput(), flags=re.I)
        self.assertTrue(match)
        return match.group(1) if match else ""
