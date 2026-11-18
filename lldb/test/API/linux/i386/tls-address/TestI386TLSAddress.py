"""Test evaluating TLS locations in a 32-bit x86 Linux inferior."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class I386TLSAddressTestCase(TestBase):
    @requireLinux
    @skipIf(archs=no_match(["x86_64", "i386", "i686"]))
    @skipUnlessCompilerSupports("-m32")
    def test(self):
        self.build(dictionary={"CFLAGS_EXTRAS": "-m32"})

        target, process, thread, breakpoint = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )
        self.assertEqual(target.GetAddressByteSize(), 4)
        self.expect("target variable tls_value", substrs=["(int) tls_value = 43"])
        self.expect("register read --all", substrs=["gs_base ="])
        self.expect(
            "register read --all",
            matching=False,
            substrs=["registers were unavailable"],
        )

        register_sets = thread.GetFrameAtIndex(0).GetRegisters()
        register_count = sum(
            register_sets[i].GetNumChildren() for i in range(register_sets.GetSize())
        )
        # Check the fallback used by clients that cannot consume target XML.
        gs_base_response = None
        for reg_index in range(register_count):
            result = lldb.SBCommandReturnObject()
            self.dbg.GetCommandInterpreter().HandleCommand(
                f"process plugin packet send qRegisterInfo{reg_index:x}", result
            )
            self.assertTrue(result.Succeeded(), result.GetError())
            if "name:gs_base;" in result.GetOutput():
                gs_base_response = result.GetOutput()
                break

        self.assertIsNotNone(gs_base_response)
        self.assertIn("generic:tp;", gs_base_response)
        self.assertIn("set:general;", gs_base_response)
