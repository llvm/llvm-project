"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibCxxFunctionTestCase(TestBase):
    # Run frame var for a variable twice. Verify we do not hit the cache
    # the first time but do the second time.
    def run_frame_var_check_cache_use(
        self, variable, result_to_match, skip_find_function=False
    ):
        self.runCmd("log timers reset")
        self.expect(
            "frame variable " + variable, substrs=[variable + " =  " + result_to_match]
        )
        if not skip_find_function:
            self.expect(
                "log timers dump", substrs=["lldb_private::CompileUnit::FindFunction"]
            )

        self.runCmd("log timers reset")
        self.expect(
            "frame variable " + variable, substrs=[variable + " =  " + result_to_match]
        )
        self.expect(
            "log timers dump",
            matching=False,
            substrs=["lldb_private::CompileUnit::FindFunction"],
        )

    @add_test_categories(["libc++"])
    def test(self):
        """Test that std::function as defined by libc++ is correctly printed by LLDB"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        bkpt = self.target().FindBreakpointByID(
            lldbutil.run_break_set_by_source_regexp(
                self, "Set break point at this line."
            )
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        self.run_frame_var_check_cache_use(
            "foo2_f", "Lambda in File main.cpp at Line 22"
        )

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.run_frame_var_check_cache_use(
            "add_num2_f", "Lambda in File main.cpp at Line 13"
        )

        lldbutil.continue_to_breakpoint(self.process(), bkpt)

        self.run_frame_var_check_cache_use("f2", "Lambda in File main.cpp at Line 35")
        self.run_frame_var_check_cache_use(
            "f3", "Lambda in File main.cpp at Line 39", True
        )
        # TODO reenable this case when std::function formatter supports
        # general callable object case.
        # self.run_frame_var_check_cache_use("f4", "Function in File main.cpp at Line 8")

        # These cases won't hit the cache at all but also don't require
        # an expensive lookup.
        self.expect("frame variable f1", substrs=["f1 =  Function = foo(int, int)"])

        self.expect(
            "frame variable f5", substrs=["f5 =  Function = Bar::add_num(int) const"]
        )
