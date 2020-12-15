"""
Tests forward declarations coming from the `std` module.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # We only emulate a fake libc++ in this test and don't use the real libc++,
    # but we still add the libc++ category so that this test is only run in
    # test configurations where libc++ is actually supposed to be tested.
    @add_test_categories(["libc++"])
    @skipIfRemote
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        sysroot = os.path.join(os.getcwd(), "root")

        # Set the sysroot where our dummy libc++ exists.
        self.runCmd("platform select --sysroot '" + sysroot + "' host",
                    CURRENT_EXECUTABLE_SET)

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        self.runCmd("settings set target.import-std-module true")

        # Print the dummy `std::vector`. It only has the dummy member in it
        # so the standard `std::vector` formatter can't format it. Instead use
        # the raw output so LLDB has to show the member variable.
        # Both `std::vector` and the type of the member have forward
        # declarations before their definitions.
        self.expect("expr --raw -- v",
                    substrs=['(std::__1::vector<int>) $0 = {', 'f = nullptr', '}'])
