"""
Test that LLDB doesn't crash if the std module we load is empty.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os

class ImportStdModule(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # We only emulate a fake libc++ in this test and don't use the real libc++,
    # but we still add the libc++ category so that this test is only run in
    # test configurations where libc++ is actually supposed to be tested.
    @add_test_categories(["libc++"])
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        sysroot = os.path.join(os.getcwd(), "root")

        # Set the sysroot.
        self.runCmd("platform select --sysroot '" + sysroot + "' host", CURRENT_EXECUTABLE_SET)

        lldbutil.run_to_source_breakpoint(self,
            "// Set break point at this line.", lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        # Use the typedef that is only defined in our 'empty' module. If this fails, then LLDB
        # somehow figured out the correct define for the header and compiled the right
        # standard module that actually contains the std::vector template.
        self.expect("expr MissingContent var = 3; var", substrs=['$0 = 3'])
        # Try to access our mock std::vector. This should fail but not crash LLDB as the
        # std::vector template should be missing from the std module.
        self.expect("expr (size_t)v.size()", substrs=["Couldn't lookup symbols"], error=True)
