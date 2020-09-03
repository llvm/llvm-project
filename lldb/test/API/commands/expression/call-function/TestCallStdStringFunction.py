"""
Test calling std::String member functions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class ExprCommandCallFunctionTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(
        compiler="icc",
        bugnumber="llvm.org/pr14437, fails with ICC 13.1")
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr21765")
    def test_with(self):
        """Test calling std::String member function."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        self.expect("print str",
                    substrs=['Hello world'])

        # Calling this function now succeeds, but we follow the typedef return type through to
        # const char *, and thus don't invoke the Summary formatter.

        # clang's libstdc++ on ios arm64 inlines std::string::c_str() always;
        # skip this part of the test.
        triple = self.dbg.GetSelectedPlatform().GetTriple()
        do_cstr_test = True
        if triple in ["arm64-apple-ios", "arm64e-apple-ios", "arm64-apple-tvos", "armv7k-apple-watchos", "arm64-apple-bridgeos", "arm64_32-apple-watchos"]:
            do_cstr_test = False
        if do_cstr_test:
            self.expect("print str.c_str()",
                        substrs=['Hello world'])
