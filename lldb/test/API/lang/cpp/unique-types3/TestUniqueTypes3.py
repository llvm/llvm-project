"""
Test that we return only the requested template instantiation.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class UniqueTypesTestCase3(TestBase):
    def do_test(self, debug_flags):
        """Test that we display the correct template instantiation."""
        self.build(dictionary=debug_flags)
        lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("a.cpp")
        )
        self.expect_expr("a", result_type="S<int>")

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_simple_template_names(self):
        # Can't directly set CFLAGS_EXTRAS here because the Makefile can't
        # override an environment variable.
        self.do_test(dict(TEST_CFLAGS_EXTRAS="-gsimple-template-names"))

    @skipIf(compiler=no_match("clang"))
    @skipIf(compiler_version=["<", "15.0"])
    def test_no_simple_template_names(self):
        self.do_test(dict(CFLAGS_EXTRAS="-gno-simple-template-names"))
