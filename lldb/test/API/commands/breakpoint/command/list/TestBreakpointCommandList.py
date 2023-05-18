"""
Test 'breakpoint command list'.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    @no_debug_info_test
    def test_list_commands(self):
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "a.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        # Test without any breakpoints.
        self.expect(
            "breakpoint command list 1",
            error=True,
            substrs=["error: No breakpoints exist for which to list commands"],
        )

        # Set a breakpoint
        self.runCmd("b foo")

        # Check list breakpoint commands for breakpoints that have no commands.
        self.expect(
            "breakpoint command list 1",
            startstr="Breakpoint 1 does not have an associated command.",
        )

        # Add a breakpoint command.
        self.runCmd("breakpoint command add -o 'source list' 1")

        # List breakpoint command that we just created.
        self.expect(
            "breakpoint command list 1",
            startstr="""Breakpoint 1:
    Breakpoint commands:
      source list
""",
        )

        # List breakpoint command with invalid breakpoint ID.
        self.expect(
            "breakpoint command list 2",
            error=True,
            startstr="error: '2' is not a currently valid breakpoint ID.",
        )
