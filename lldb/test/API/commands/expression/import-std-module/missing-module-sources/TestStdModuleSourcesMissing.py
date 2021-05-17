"""
Check that missing module source files are correctly handled by LLDB.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os
import shutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # We only emulate a fake libc++ in this test and don't use the real libc++,
    # but we still add the libc++ category so that this test is only run in
    # test configurations where libc++ is actually supposed to be tested.
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        # The path to our temporary target root that contains the temporary
        # module sources.
        target_sysroot = self.getBuildArtifact("root")

        # Copy the sources to the root.
        shutil.copytree(self.getSourcePath("root"), target_sysroot)
        # Build the binary with the copied sources.
        self.build()
        # Delete the copied sources so that they are now unavailable.
        shutil.rmtree(target_sysroot)

        # Set the sysroot where our dummy libc++ used to exist. Just to make
        # sure we don't find some existing headers on the system that could
        # XPASS this test.
        self.runCmd("platform select --sysroot '" + target_sysroot + "' host")

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        # Import the std C++ module and run an expression.
        # As we deleted the sources, LLDB should refuse the load the module
        # and just print the normal error we get from the expression.
        self.runCmd("settings set target.import-std-module true")
        self.expect("expr v.unknown_identifier", error=True,
                    substrs=["no member named 'unknown_identifier'"])
        # Check that there is no confusing error about failing to build the
        # module.
        self.expect("expr v.unknown_identifier", error=True, matching=False,
                    substrs=["could not build module 'std'"])

        # Test the fallback mode. It should also just print the normal
        # error but not mention a failed module build.
        self.runCmd("settings set target.import-std-module fallback")

        self.expect("expr v.unknown_identifier", error=True,
                     substrs=["no member named 'unknown_identifier'"])
        self.expect("expr v.unknown_identifier", error=True, matching=False,
                    substrs=["could not build module 'std'"])
