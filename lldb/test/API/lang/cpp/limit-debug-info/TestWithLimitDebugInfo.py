import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWithLimitDebugInfo(TestBase):
    SHARED_BUILD_TESTCASE = False

    def _run_test(self, build_dict):
        self.build(dictionary=build_dict)

        # Break on main function
        _, _, _, bkpt = lldbutil.run_to_line_breakpoint(
            self,
            lldb.SBFileSpec("derived.h", False),
            line_number("derived.h", "// break1"),
        )
        self.assertEqual(bkpt.GetNumLocations(), 1)
        lldbutil.run_break_set_by_file_and_line(
            self, "derived.h", line_number("derived.h", "// break2")
        )

        self.expect_expr("1", result_type="int", result_value="1")
        self.expect_expr("this", result_type="Foo *")
        self.expect_expr("this->x", result_type="int", result_value="12345")

        self.runCmd("continue")

        self.expect_expr("1", result_type="int", result_value="1")
        self.expect_expr("this", result_type="ns::Foo2 *")
        self.expect_expr("this->x", result_type="int", result_value="23456")

    @add_test_categories(["dwarf", "dwo"])
    def test_default(self):
        self._run_test(dict(CFLAGS_EXTRAS="$(LIMIT_DEBUG_INFO_FLAGS)"))

    @add_test_categories(["dwarf", "dwo"])
    def test_debug_names(self):
        self._run_test(
            dict(CFLAGS_EXTRAS="$(LIMIT_DEBUG_INFO_FLAGS) -gdwarf-5 -gpubnames")
        )
