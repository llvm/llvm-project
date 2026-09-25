"""
Test lldb data formatter subsystem for std::vector<bool>.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdVBoolDataFormatterTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    def do_test(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.max-children-count 128")
        self.addTearDownHook(
            lambda: self.runCmd("settings set target.max-children-count 24")
        )

        self.expect("frame variable vBoolEmpty", substrs=["size=0"])
        self.expect("expr -- vBoolEmpty", substrs=["size=0"])

        expected_small = [
            "size=10",
            "[0] = true",
            "[1] = false",
            "[2] = true",
            "[3] = true",
            "[4] = false",
            "[5] = false",
            "[6] = true",
            "[7] = false",
            "[8] = true",
            "[9] = true",
        ]
        self.expect("frame variable vBoolSmall", substrs=expected_small)
        self.expect("expr -- vBoolSmall", substrs=expected_small)

        expected = [
            "size=73",
            "[0] = false",
            "[1] = true",
            "[18] = false",
            "[27] = true",
            "[36] = false",
            "[47] = true",
            "[48] = true",
            "[49] = true",
            "[50] = false",
            "[56] = false",
            "[65] = true",
            "[70] = false",
            "[71] = true",
            "[72] = true",
        ]
        self.expect("frame variable vBool", substrs=expected)
        self.expect("expr -- vBool", substrs=expected)

    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["libstdcxx"])
    def test_libstdcxx_debug(self):
        self.build(
            dictionary={"USE_LIBSTDCPP": 1, "CXXFLAGS_EXTRAS": "-D_GLIBCXX_DEBUG"}
        )
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test()
