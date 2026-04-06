import os
import shutil
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


def has_lldb_codesign():
    """Check if the lldb_codesign certificate is available."""
    try:
        result = subprocess.run(
            [
                "security",
                "find-certificate",
                "-c",
                "lldb_codesign",
                "/Library/Keychains/System.keychain",
            ],
            capture_output=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


@skipUnlessDarwin
class TestdSYMCodesign(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def build_dsym_with_script(self):
        self.build(debug_info="dsym")
        exe = self.getBuildArtifact("a.out")
        dsym = self.getBuildArtifact("a.out.dSYM")
        python_dir = os.path.join(dsym, "Contents", "Resources", "Python")
        os.makedirs(python_dir, exist_ok=True)
        shutil.copy(
            os.path.join(self.getSourceDir(), "dsym_script.py"),
            os.path.join(python_dir, "a.py"),
        )
        return exe, dsym

    def test_adhoc_signed_dsym(self):
        """An ad-hoc signed dSYM should not be loaded because the
        signature doesn't chain to a trusted root CA."""
        exe, dsym = self.build_dsym_with_script()
        subprocess.check_call(["codesign", "-f", "-s", "-", dsym])

        self.runCmd("settings set target.load-script-from-symbol-file trusted")
        self.createTestTarget(file_path=exe)

        self.expect(
            "script -- print('SENTINEL')",
            substrs=["SENTINEL"],
        )
        # The script should NOT have been loaded.
        self.assertFalse(
            hasattr(lldb, "_dsym_codesign_test_loaded"),
            "Script should not auto-load from ad-hoc signed dSYM",
        )

    @unittest.skipUnless(has_lldb_codesign(), "requires lldb_codesign certificate")
    def test_trusted_signed_dsym_auto_loads(self):
        """A dSYM signed with the trusted lldb_codesign certificate should
        auto-load scripts."""
        exe, dsym = self.build_dsym_with_script()
        subprocess.check_call(["codesign", "-f", "-s", "lldb_codesign", dsym])

        self.runCmd("settings set target.load-script-from-symbol-file trusted")
        self.createTestTarget(file_path=exe)

        # The script sets a marker attribute on the lldb module.
        self.assertTrue(
            getattr(lldb, "_dsym_codesign_test_loaded", False),
            "Script should auto-load from trusted signed dSYM",
        )
