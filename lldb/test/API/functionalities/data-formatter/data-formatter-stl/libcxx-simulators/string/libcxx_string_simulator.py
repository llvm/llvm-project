"""
Shared test logic for checking that we can understand various layouts of
libc++'s std::string. Each TestLibcxxStringRev<N>.py file tests one revision.
"""

import lldb
from lldbsuite.test import lldbutil
import functools


def _run_test(self, defines):
    cxxflags_extras = " ".join(["-D%s" % d for d in defines])
    self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxxflags_extras))
    lldbutil.run_to_source_breakpoint(
        self, "// Break here", lldb.SBFileSpec("main.cpp")
    )
    self.expect_var_path("shortstring", summary='"short"')
    self.expect_var_path("longstring", summary='"I am a very long string"')

    self.expect_expr("shortstring", result_summary='"short"')
    self.expect_expr("longstring", result_summary='"I am a very long string"')


def add_tests(test_case, revision):
    """Adds a test method to test_case for each layout variant of the given
    std::string revision."""
    for v in [None, "ALTERNATE_LAYOUT"]:
        for c in range(5):
            name = "test_r%d_c%d" % (revision, c)
            defines = ["REVISION=%d" % revision, "COMPRESSED_PAIR_REV=%d" % c]
            if v:
                name += "_" + v
                defines += [v]

            @functools.wraps(_run_test)
            def test_method(self, defines=defines):
                _run_test(self, defines)

            test_method.__name__ = name
            setattr(test_case, name, test_method)
