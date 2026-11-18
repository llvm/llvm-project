"""Test that pointers with authentication bits are symbolicated using their
stripped values."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbarm64e import Arm64eTestBase


class TestPointerAuthStripping(Arm64eTestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def test_strip_authentication_bits(self):
        self.build()
        _, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c", False)
        )
        frame = thread.GetFrameAtIndex(0)

        # lr holds the signed address in main() that foo() returns to.
        signed_lr = frame.FindRegister("lr").GetValueAsUnsigned()
        stripped_lr = process.FixAddress(signed_lr)
        self.assertNotEqual(signed_lr, stripped_lr)
        self.expect("register read lr", substrs=["(0x%016x)" % stripped_lr, "`main"])
        # pc resolves without stripping, so it prints no stripped address.
        self.expect("register read pc", substrs=["(0x"], matching=False)
        self.expect("image lookup -a $lr", substrs=["`main"])

        # main_ptr is a signed function pointer to main().
        main_ptr = frame.FindVariable("main_ptr")
        self.assertTrue(main_ptr.IsValid())
        signed_addr = main_ptr.GetValueAsUnsigned(lldb.LLDB_INVALID_ADDRESS)
        stripped_addr = main_ptr.GetValueAsAddress()
        self.assertNotEqual(signed_addr, lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(stripped_addr, lldb.LLDB_INVALID_ADDRESS)
        self.assertNotEqual(signed_addr, stripped_addr)

        stream = lldb.SBStream()
        main_ptr.GetDescription(stream)
        self.assertIn("`main", stream.GetData())
        self.expect("p main_ptr", substrs=["`main"])

        # Store the signed pointer on the stack and read it back.
        self.runCmd("p *(uint64_t *)$sp = *(uint64_t *)&main_ptr")
        self.expect("memory read -f A -c 1 $sp", substrs=["`main"])
