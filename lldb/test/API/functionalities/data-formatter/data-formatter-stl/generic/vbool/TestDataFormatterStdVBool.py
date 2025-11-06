"""
Test lldb data formatter subsystem.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdVBoolDataFormatterTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def do_test(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)
            self.runCmd("settings set target.max-children-count 24", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            "frame variable -A vBool",
            substrs=[
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
            ],
        )

        self.expect(
            "expr -A -- vBool",
            substrs=[
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
            ],
        )

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
    def test_libstdcxx(self):
        # No flags, because the "msvcstl" category checks that the MSVC STL is used by default.
        self.build()
        self.do_test()
