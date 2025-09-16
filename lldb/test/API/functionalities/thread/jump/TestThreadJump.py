"""
Test jumping to different places.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ThreadJumpTestCase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.build()

    def test(self):
        """Test thread jump handling."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Find the line numbers for our breakpoints.
        self.mark1 = line_number("main.cpp", "// 1st marker")
        self.mark2 = line_number("main.cpp", "// 2nd marker")
        self.mark3 = line_number("main.cpp", "// 3rd marker")
        self.mark4 = line_number("main.cpp", "// 4th marker")
        self.mark5 = line_number("other.cpp", "// other marker")

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.mark3, num_expected_locations=1
        )
        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint 1.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT + " 1",
            substrs=[
                "stopped",
                "main.cpp:{}".format(self.mark3),
                "stop reason = breakpoint 1",
            ],
        )

        # Try the int path, force it to return 'a'
        self.do_min_test(self.mark3, self.mark1, "i", "4")
        # Try the int path, force it to return 'b'
        self.do_min_test(self.mark3, self.mark2, "i", "5")
        # Try the double path, force it to return 'a'
        self.do_min_test(self.mark4, self.mark1, "j", "7")
        # Expected to fail on powerpc64le architecture
        if not self.isPPC64le():
            # Try the double path, force it to return 'b'
            self.do_min_test(self.mark4, self.mark2, "j", "8")

        # Try jumping to another function in a different file.
        self.runCmd("thread jump --file other.cpp --line %i --force" % self.mark5)
        self.expect("process status", substrs=["at other.cpp:%i" % self.mark5])

        # Try jumping to another function (without forcing)
        self.expect(
            "j main.cpp:%i" % self.mark1,
            COMMAND_FAILED_AS_EXPECTED,
            error=True,
            substrs=["error"],
        )

    @expectedFailureAll(compiler="clang", compiler_version=["<", "17.0"])
    def test_jump_offset(self):
        """Test Thread Jump by negative or positive offset"""
        exe = self.getBuildArtifact("a.out")
        file_name = "main.cpp"
        self.runCmd(f"target create {exe}", CURRENT_EXECUTABLE_SET)

        pos_jump = line_number(file_name, "// jump_offset 1")
        neg_jump = line_number(file_name, "// jump_offset 2")
        pos_breakpoint = line_number(file_name, "// breakpoint 1")
        neg_breakpoint = line_number(file_name, "// breakpoint 2")
        pos_jump_offset = pos_jump - pos_breakpoint
        neg_jump_offset = neg_jump - neg_breakpoint

        var_1, var_1_value = ("var_1", "10")
        var_2, var_2_value = ("var_2", "40")
        var_3, var_3_value = ("var_3", "10")

        # create pos_breakpoint and neg_breakpoint
        lldbutil.run_break_set_by_file_and_line(
            self, file_name, pos_breakpoint, num_expected_locations=1
        )
        lldbutil.run_break_set_by_file_and_line(
            self, file_name, neg_breakpoint, num_expected_locations=1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # test positive jump
        # The stop reason of the thread should be breakpoint 1.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT + " 1",
            substrs=[
                "stopped",
                f"{file_name}:{pos_breakpoint}",
                "stop reason = breakpoint 1",
            ],
        )

        self.runCmd(f"thread jump --by +{pos_jump_offset}")
        self.expect("process status", substrs=[f"at {file_name}:{pos_jump}"])
        self.expect(f"print {var_1}", substrs=[var_1_value])

        self.runCmd("thread step-over")
        self.expect(f"print {var_2}", substrs=[var_2_value])

        self.runCmd("continue")

        # test negative jump
        # The stop reason of the thread should be breakpoint 1.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT + " 2",
            substrs=[
                "stopped",
                f"{file_name}:{neg_breakpoint}",
                "stop reason = breakpoint 2",
            ],
        )

        self.runCmd(f"thread jump --by {neg_jump_offset}")
        self.expect("process status", substrs=[f"at {file_name}:{neg_jump}"])
        self.runCmd("thread step-over")
        self.expect(f"print {var_3}", substrs=[var_3_value])

    def do_min_test(self, start, jump, var, value):
        # jump to the start marker
        self.runCmd("j %i" % start)
        self.runCmd("thread step-in")  # step into the min fn
        # jump to the branch we're interested in
        self.runCmd("j %i" % jump)
        self.runCmd("thread step-out")  # return out
        self.runCmd("thread step-over")  # assign to the global
        self.expect("expr %s" % var, substrs=[value])  # check it
