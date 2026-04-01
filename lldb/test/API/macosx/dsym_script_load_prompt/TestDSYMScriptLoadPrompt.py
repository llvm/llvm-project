"""
Test the interactive prompt when target.load-script-from-symbol-file
is set to 'warn' and a dSYM contains a Python script.
"""

import os
import shutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


@skipUnlessDarwin
class TestDSYMScriptLoadPrompt(PExpectTest):
    def build_with_dsym_script(self):
        self.build(debug_info="dsym")
        exe = self.getBuildArtifact("a.out")
        dsym_path = exe + ".dSYM"
        python_dir = os.path.join(dsym_path, "Contents", "Resources", "Python")
        os.makedirs(python_dir, exist_ok=True)
        shutil.copy(
            os.path.join(self.getSourceDir(), "dsym_script.py"),
            os.path.join(python_dir, "a.py"),
        )
        return exe

    def test_prompt_accept(self):
        """Test that replying 'Y' to the prompt loads the script."""
        exe = self.build_with_dsym_script()

        self.launch(
            extra_args=[
                "-O",
                "settings set target.load-script-from-symbol-file warn",
            ]
        )
        self.child.sendline("target create " + exe)
        self.child.expect_exact("To run this script in this debug")
        self.child.expect_exact("Do you want to load the script")
        self.child.expect_exact("a.py")
        self.child.sendline("Y")
        self.expect_prompt()
        self.child.sendline("script lldb.LOADED")
        self.child.expect_exact("True")
        self.expect_prompt()
        self.quit()

    def test_prompt_deny(self):
        """Test that replying 'N' to the prompt does not load the script."""
        exe = self.build_with_dsym_script()

        self.launch(
            extra_args=[
                "-O",
                "settings set target.load-script-from-symbol-file warn",
            ]
        )
        self.child.sendline("target create " + exe)
        self.child.expect_exact("To run this script in this debug")
        self.child.expect_exact("Do you want to load the script")
        self.child.expect_exact("a.py")
        self.child.sendline("N")
        self.expect_prompt()
        self.child.sendline("script lldb.LOADED")
        self.child.expect_exact("module 'lldb' has no attribute 'LOADED'")
        self.quit()

    def test_prompt_default_is_no(self):
        """Test that the default reply (just pressing 'Enter') is 'N'."""
        exe = self.build_with_dsym_script()

        self.launch(
            extra_args=[
                "-O",
                "settings set target.load-script-from-symbol-file warn",
            ]
        )
        self.child.sendline("target create " + exe)
        self.child.expect_exact("To run this script in this debug")
        self.child.expect_exact("Do you want to load the script")
        self.child.expect_exact("a.py")
        self.child.sendline("")
        self.expect_prompt()
        self.child.sendline("script lldb.LOADED")
        self.child.expect_exact("module 'lldb' has no attribute 'LOADED'")
        self.quit()
