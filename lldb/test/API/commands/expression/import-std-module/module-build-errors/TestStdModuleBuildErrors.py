"""
Tests that the import-std-module=fallback setting is only showing the error
diagnostics from the first parse attempt which isn't using a module.
This is supposed to prevent that a broken libc++ module renders failing
expressions useless as the original failing errors are suppressed by the
module build errors.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import os


class TestCase(TestBase):

    # We only emulate a fake libc++ in this test and don't use the real libc++,
    # but we still add the libc++ category so that this test is only run in
    # test configurations where libc++ is actually supposed to be tested.
    @add_test_categories(["libc++"])
    @skipIfRemote
    @skipIf(compiler=no_match("clang"))
    def test(self):
        self.build()

        sysroot = os.path.join(os.getcwd(), "root")

        # Set the sysroot this test is using to provide a custom libc++.
        self.runCmd("platform select --sysroot '" + sysroot + "' host",
                    CURRENT_EXECUTABLE_SET)

        lldbutil.run_to_source_breakpoint(self,
                                          "// Set break point at this line.",
                                          lldb.SBFileSpec("main.cpp"))

        # The expected error message when the fake libc++ module in this test
        # fails to build from within LLDB (as it contains invalid code).
        module_build_error_msg = "unknown type name 'random_token_to_fail_the_build'"

        # First force the std module to be imported. This should show the
        # module build error to the user.
        self.runCmd("settings set target.import-std-module true")
        self.expect("expr (size_t)v.size()",
                    substrs=[module_build_error_msg],
                    error=True)

        # In the fallback mode the module build error should not be shown.
        self.runCmd("settings set target.import-std-module fallback")
        fallback_expr = "expr v ; error_to_trigger_fallback_mode"
        # First check for the actual expression error that should be displayed
        # and is useful for the user.
        self.expect(fallback_expr,
                    substrs=["use of undeclared identifier 'error_to_trigger_fallback_mode'"],
                    error=True)
        # Test that the module build error is not displayed.
        self.expect(fallback_expr,
                    substrs=[module_build_error_msg],
                    matching=False,
                    error=True)
