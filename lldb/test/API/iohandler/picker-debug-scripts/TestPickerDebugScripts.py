"""
Test picking which debug scripts to load from a picker.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


@skipIfAsan
@skipIfRemote
@skipIfWindows
class PickerDebugScriptsTest(PExpectTest):
    def write_script(self, safe_path, module):
        # `testing.safe-auto-load-paths` makes LLDB find "libalpha/libalpha.py"
        # for "libalpha.so" or "libalpha.dylib".
        script_dir = os.path.join(safe_path, module)
        os.makedirs(script_dir, exist_ok=True)
        with open(os.path.join(script_dir, module + ".py"), "w") as script:
            script.write(
                "def __lldb_init_module(debugger, internal_dict):\n"
                f"    print('{module.upper()}_LOADED')\n"
            )

    def test(self):
        """Debug scripts from three libraries, one of them trusted, are listed
        in one picker, once."""
        self.build()
        if not self.dbg.GetCommandInterpreter().CommandExists("test-picker"):
            self.skipTest("testing settings require a build with assertions")

        safe_path = self.getBuildArtifact("safe-path")
        for module in ["libalpha", "libbeta", "libgamma"]:
            self.write_script(safe_path, module)

        self.launch(dimensions=(24, 120))
        # launch() runs "settings clear --all", which resets the terminal size.
        self.expect("settings set term-height 24")
        self.expect("settings set term-width 120")
        exe = self.getBuildArtifact("a.out")
        # Without dependents, the libraries load together when the process
        # launches.
        self.expect(f"target create --no-dependents {exe}", ["Current executable"])
        self.expect("settings set target.load-script-from-symbol-file warn")
        self.expect(f"settings append testing.safe-auto-load-paths {safe_path}")
        self.expect("settings append testing.trusted-auto-load-modules libalpha")
        self.expect("breakpoint set -p 'break here'", ["Breakpoint 1"])

        self.child.sendline("run")
        self.child.expect_exact("Select debug scripts to load:")
        # Only the trusted script starts checked.
        self.child.expect_exact("1/3 selected")
        self.child.send("\r")
        self.child.expect_exact("LIBALPHA_LOADED")
        self.expect_prompt()
        self.assertNotIn(b"LIBBETA_LOADED", self.child.before)
        self.assertNotIn(b"LIBGAMMA_LOADED", self.child.before)

        # Relaunching loads the libraries again, which mustn't offer the same
        # scripts twice.
        self.expect("process kill")
        self.expect("run", ["stop reason = breakpoint 1"])
        self.assertNotIn(b"Select debug scripts", self.child.before)
        self.quit()
