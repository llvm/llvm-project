from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestWithLimitDebugInfo(TestBase):
    SHARED_BUILD_TESTCASE = False

    def _run_test(self, build_dict):
        self.build(dictionary=build_dict)

        # Get the path of the executable
        exe_path = self.getBuildArtifact("a.out")

        # Load the executable
        target = self.dbg.CreateTarget(exe_path)
        self.assertTrue(target.IsValid(), VALID_TARGET)

        # Break on main function
        break1_id = lldbutil.run_break_set_by_file_and_line(
            self, "derived.h", line_number("derived.h", "// break1")
        )
        lldbutil.run_break_set_by_file_and_line(
            self, "derived.h", line_number("derived.h", "// break2")
        )

        lldbutil.run_to_breakpoint_do_run(
            self, target, target.FindBreakpointByID(break1_id)
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
