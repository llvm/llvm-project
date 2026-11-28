"""Test that the SB API follows __ptrauth-qualified data pointers."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test.lldbarm64e import Arm64eTestBase


class TestPtrAuthSBValue(Arm64eTestBase):
    NO_DEBUG_INFO_TESTCASE = True
    SHARED_BUILD_TESTCASE = False

    def test_follow_ptrauth_sbvalue_elems(self):
        """Test following ptrauth qualified data pointers to children through SB API"""
        self.build()
        target, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.c")
        )

        frame = thread.GetFrameAtIndex(0)

        # A variable path follows authenticated pointers through the chain.
        value = frame.GetValueForVariablePath(
            "two.auth_middle->auth_bottom->auth_valuep"
        ).Dereference()
        self.assertEqual(value.GetValueAsUnsigned(), 2)

        top = frame.FindVariable("two")
        self.assertTrue(top.IsValid())

        unauth_middle_p = top.GetChildMemberWithName("unauth_middle")
        self.assertTrue(unauth_middle_p.IsValid())

        auth_middle_p = top.GetChildMemberWithName("auth_middle")
        self.assertTrue(auth_middle_p.IsValid())

        # Both pointers hold the same address, but only one of them is signed.
        self.assertEqual(
            unauth_middle_p.GetValueAsAddress(), auth_middle_p.GetValueAsAddress()
        )
        self.assertNotEqual(
            unauth_middle_p.GetValueAsUnsigned(), auth_middle_p.GetValueAsUnsigned()
        )

        unauth_middle = unauth_middle_p.Dereference()
        auth_middle = auth_middle_p.Dereference()
        self.assertEqual(auth_middle.GetLoadAddress(), unauth_middle.GetLoadAddress())

        unauth_bottom_p = auth_middle.GetChildMemberWithName("unauth_bottom")
        auth_bottom_p = unauth_middle.GetChildMemberWithName("auth_bottom")

        self.assertEqual(
            unauth_bottom_p.GetValueAsAddress(), auth_bottom_p.GetValueAsAddress()
        )
        self.assertNotEqual(
            unauth_bottom_p.GetValueAsUnsigned(), auth_bottom_p.GetValueAsUnsigned()
        )

        unauth_bottom = unauth_bottom_p.Dereference()
        auth_bottom = auth_bottom_p.Dereference()
        self.assertEqual(auth_bottom.GetLoadAddress(), unauth_bottom.GetLoadAddress())

        self.assertEqual(
            auth_bottom.GetChildMemberWithName("value").GetValueAsUnsigned(), 1
        )

        unauth_valuep = auth_bottom.GetChildMemberWithName("unauth_valuep")
        auth_valuep = auth_bottom.GetChildMemberWithName("auth_valuep")

        self.assertEqual(
            unauth_valuep.GetValueAsAddress(), auth_valuep.GetValueAsAddress()
        )
        self.assertNotEqual(
            unauth_valuep.GetValueAsUnsigned(), auth_valuep.GetValueAsUnsigned()
        )

        self.assertEqual(unauth_valuep.Dereference().GetValueAsUnsigned(), 2)
        self.assertEqual(auth_valuep.Dereference().GetValueAsUnsigned(), 2)

        value = frame.GetValueForVariablePath(
            "*two.auth_middle->auth_bottom->auth_valuep"
        )
        self.assertEqual(value.GetValueAsUnsigned(), 2)
        stack_val = frame.FindVariable("stack_val")
        self.assertEqual(value.GetLoadAddress(), stack_val.GetLoadAddress())

        # Create a value from an address that carries signature bits.
        int_sbtype = target.FindFirstType("int")
        addr_with_ptrauth_bits = lldb.SBAddress()
        addr_with_ptrauth_bits.SetLoadAddress(auth_valuep.GetValueAsUnsigned(), target)
        intptr_from_addr = target.CreateValueFromAddress(
            "test1", addr_with_ptrauth_bits, int_sbtype
        )
        self.assertTrue(intptr_from_addr.IsValid())
        self.assertEqual(intptr_from_addr.GetValueAsUnsigned(), 2)
        self.assertEqual(intptr_from_addr.GetLoadAddress(), stack_val.GetLoadAddress())
