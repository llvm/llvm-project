"""
This is a sanity check that verifies that the module cache path is set
correctly and points inside the default test build directory.
"""


import sys

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class ModuleCacheSanityTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        build_dir_name = (
            "lldb-test-build" if sys.platform == "win32" else "lldb-test-build.noindex"
        )
        self.expect(
            "settings show symbols.clang-modules-cache-path",
            substrs=[build_dir_name, "module-cache-lldb"],
        )
