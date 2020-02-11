"""Test that lldb works correctly on compile units form different languages."""



import re
import lldb
from lldbsuite.test.lldbtest import *


class MixedLanguagesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_language_of_frame(self):
        """Test that the language defaults to the language of the current frame."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Execute the cleanup function during test case tear down
        # to restore the frame format.
        def cleanup():
            self.runCmd(
                "settings set frame-format %s" %
                self.format_string, check=False)
        self.addTearDownHook(cleanup)
        self.runCmd("settings show frame-format")
        m = re.match(
            '^frame-format \(format-string\) = "(.*)\"$',
            self.res.GetOutput())
        self.assertTrue(m, "Bad settings string")
        self.format_string = m.group(1)

        # Change the default format to print the language.
        format_string = "frame #${frame.index}: ${frame.pc}{ ${module.file.basename}\`${function.name}{${function.pc-offset}}}{, lang=${language}}\n"
        self.runCmd("settings set frame-format %s" % format_string)
        self.expect("settings show frame-format", SETTING_MSG("frame-format"),
                    substrs=[format_string])

        # Run to BP at main (in main.c) and test that the language is C.
        self.runCmd("breakpoint set -n main")
        self.runCmd("run")
        self.expect("thread backtrace",
                    substrs=["`main", "lang=c"])
        # Make sure evaluation of C++11 fails.
        self.expect("expr foo != nullptr", error=True,
                    startstr="error")

        # Run to BP at foo (in foo.cpp) and test that the language is C++.
        self.runCmd("breakpoint set -n foo")
        self.runCmd("continue")
        self.expect("thread backtrace",
                    substrs=["`::foo()", "lang=c++"])
        # Make sure we can evaluate an expression requiring C++11
        # (note: C++11 is enabled by default for C++).
        self.expect("expr foo != nullptr",
                    patterns=["true"])
