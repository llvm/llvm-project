"""
Test that addresses of functions compiled for Arm Thumb include the Thumb mode
bit (bit 0 of the address) when resolved and used in expressions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestThumbFunctionAddr(TestBase):
    def do_thumb_function_test(self, language):
        self.build(dictionary={"CFLAGS_EXTRAS": f"-x {language} -mthumb"})

        exe = self.getBuildArtifact("a.out")
        line = line_number("main.c", "// Set break point at this line.")
        self.runCmd("target create %s" % exe)
        bpid = lldbutil.run_break_set_by_file_and_line(self, "main.c", line)

        self.runCmd("run")
        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint_id(self.process(), bpid),
            "Process is not stopped at breakpoint",
        )

        # The compiler set this, so the mode bit will be included here.
        a_function_addr_var = (
            self.thread().GetFrameAtIndex(0).FindVariable("a_function_addr")
        )
        self.assertTrue(a_function_addr_var.IsValid())
        a_function_addr = a_function_addr_var.GetValueAsUnsigned()
        self.assertTrue(a_function_addr & 1)

        self.expect("p/x a_function_addr", substrs=[f"0x{a_function_addr:08x}"])
        # If lldb did not pay attention to the mode bit this would SIGILL trying
        # to execute Thumb encodings in Arm mode.
        self.expect("expression -- a_function()", substrs=["= 123"])

        # We cannot call GetCallableLoadAdress via. the API, so we expect this
        # to not have the bit set as it's treating it as a non-function symbol.
        found_function = self.target().FindFunctions("a_function")[0]
        self.assertTrue(found_function.IsValid())
        found_function = found_function.GetFunction()
        self.assertTrue(found_function.IsValid())
        found_function_addr = found_function.GetStartAddress()
        a_function_load_addr = found_function_addr.GetLoadAddress(self.target())
        self.assertEqual(a_function_load_addr, a_function_addr & ~1)

        # image lookup should not include the mode bit.
        a_function_file_addr = found_function_addr.GetFileAddress()
        self.expect(
            "image lookup -n a_function", substrs=[f"0x{a_function_file_addr:08x}"]
        )

    # This test is run for C and C++ because the two will take different paths
    # trying to resolve the function's address.

    @skipIf(archs=no_match(["arm$"]))
    @skipIf(archs=["arm64"])
    def test_function_addr_c(self):
        self.do_thumb_function_test("c")

    @skipIf(archs=no_match(["arm$"]))
    @skipIf(archs=["arm64"])
    def test_function_addr_cpp(self):
        self.do_thumb_function_test("c++")
