"""
Test that pending breakpoints resolve for JITted code with mcjit and rtdyld.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import configuration


class TestJitBreakpoint(TestBase):
    @skipUnlessArch("x86_64")
    @skipUnlessCompilerIsClang
    @expectedFailureAll(oslist=["windows"])
    def test_jit_breakpoints(self):
        self.build()
        self.ll = self.getBuildArtifact("jitbp.ll")
        self.do_test("--jit-kind=mcjit")
        self.do_test("--jit-linker=rtdyld")

    def do_test(self, jit_flag: str):
        self.runCmd("settings set plugin.jit-loader.gdb.enable on")

        self.assertIsNotNone(
            configuration.llvm_tools_dir,
            "llvm_tools_dir must be set to find lli",
        )
        lli_path = os.path.join(os.path.join(configuration.llvm_tools_dir, "lli"))
        self.assertTrue(lldbutil.is_exe(lli_path), f"'{lli_path}' is not an executable")
        self.runCmd(f"target create {lli_path}", CURRENT_EXECUTABLE_SET)

        line = line_number("jitbp.cpp", "int jitbp()")
        lldbutil.run_break_set_by_file_and_line(
            self, "jitbp.cpp", line, num_expected_locations=0
        )

        self.runCmd(f"run {jit_flag} {self.ll}", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        # And it should break at jitbp.cpp:1.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=[
                "stopped",
                "jitbp.cpp:%d" % line,
                "stop reason = breakpoint",
            ],
        )
