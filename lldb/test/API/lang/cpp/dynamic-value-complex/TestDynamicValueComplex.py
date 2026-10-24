"""
Test that we can resolve dynamic values in complex inheritance hierarchies.
This is mainly relevant for the Microsoft ABI where there can be multiple vtables in one type.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class DynamicValueComplexTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False

    def test_rtti(self):
        """Test with RTTI"""
        self.build(dictionary={"CXXFLAGS_EXTRAS": ""})
        self.do_test(True)

    def test_no_rtti(self):
        """Test without RTTI"""
        self.build(dictionary={"CXXFLAGS_EXTRAS": "-fno-rtti"})
        self.do_test(False)

    def do_test(self, has_rtti: bool):
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp")
        )

        frame = thread.GetFrameAtIndex(0)
        get = lambda name: self._check_get(frame, name)

        is_windows_msvc = self.getPlatform() == "windows" and "-msvc" in (
            frame.GetModule().GetTriple() or ""
        )

        # Test 'Single'.
        # struct Single : Base1
        single = get("single")
        single_base1 = get("single_base1")
        single_addr = single.GetValueAsAddress()
        self.assertNotEqual(single_addr, lldb.LLDB_INVALID_ADDRESS)
        self.assertEqual(single_addr, single_base1.GetValueAsAddress())
        self.assertEqual(single.GetTypeName(), single_base1.GetTypeName())

        single_children = [
            ValueCheck(
                type="Base1",
                children=[ValueCheck(type="int", name="base1", value="1")],
            ),
            ValueCheck(type="int", name="single", value="3"),
        ]
        self.expect_expr(
            "single", result_type="Single *", result_children=single_children
        )

        # Test 'Multiple'.
        # struct Multiple : virtual VBase, Base1, Base2
        multi = get("multi")
        multi_base1 = get("multi_base1")
        multi_base2 = get("multi_base2")
        multi_addr = multi.GetValueAsAddress()
        self.assertNotEqual(multi_addr, lldb.LLDB_INVALID_ADDRESS)
        self.assertEqual(multi_addr, multi_base1.GetValueAsAddress())
        self.assertEqual(multi_addr, multi_base2.GetValueAsAddress())
        self.assertEqual(multi.GetTypeName(), multi_base1.GetTypeName())
        self.assertEqual(multi.GetTypeName(), multi_base2.GetTypeName())

        multi_children = [
            ValueCheck(
                type="VBase",
                children=[ValueCheck(type="int", name="vbase", value="5")],
            ),
            ValueCheck(
                type="Base1",
                children=[ValueCheck(type="int", name="base1", value="1")],
            ),
            ValueCheck(
                type="Base2",
                children=[ValueCheck(type="int", name="base2", value="2")],
            ),
            ValueCheck(type="int", name="multiple", value="4"),
        ]

        # The index of 'VBase' in 'Multiple' depends on the ABI.
        # On Itanium, it sits at the start whereas on Microsoft it's after
        # the non-virtual ones.
        multiple_type = self.target().FindFirstType("Multiple")
        self.assertTrue(multiple_type.IsValid())
        first_base = multiple_type.GetDirectBaseClassAtIndex(0).GetName()
        vbases_at_start = first_base == "VBase"
        if not vbases_at_start:
            # Move 'VBase' after 'Base2'.
            self.assertEqual(first_base, "Base1")
            multi_children.insert(2, multi_children.pop(0))

        self.expect_expr(
            "multi", result_type="Multiple *", result_children=multi_children
        )

        # Test 'VBaseDiamondUser'.
        # struct VBaseDiamondUser : VBaseUser1, VBaseUser2
        vbuser = get("vbuser")
        vbuser_vbase = get("vbuser_vbase")
        vbuser_addr = vbuser.GetValueAsAddress()
        if is_windows_msvc and not has_rtti:
            # On Windows, we have to use the AST to get the offset.
            # But we don't know where the virtual base will be located in the object.
            self.assertNotEqual(vbuser_addr, lldb.LLDB_INVALID_ADDRESS)
            self.assertNotEqual(
                vbuser_vbase.GetValueAsAddress(), lldb.LLDB_INVALID_ADDRESS
            )
            self.assertNotEqual(vbuser_addr, vbuser_vbase.GetValueAsAddress())
            self.assertNotEqual(vbuser.GetTypeName(), vbuser_vbase.GetTypeName())
        else:
            self.assertNotEqual(vbuser_addr, lldb.LLDB_INVALID_ADDRESS)
            self.assertEqual(vbuser_addr, vbuser_vbase.GetValueAsAddress())
            self.assertEqual(vbuser.GetTypeName(), vbuser_vbase.GetTypeName())

        vbuser_children = [
            ValueCheck(
                type="VBaseUser1",
                children=[
                    ValueCheck(
                        type="VBase",
                        children=[ValueCheck(type="int", name="vbase", value="5")],
                    ),
                    ValueCheck(type="int", name="vbase_user1", value="6"),
                ],
            ),
            ValueCheck(
                type="VBaseUser2",
                children=[
                    ValueCheck(
                        type="VBase",
                        children=[ValueCheck(type="int", name="vbase", value="5")],
                    ),
                    ValueCheck(type="int", name="vbase_user2", value="7"),
                ],
            ),
            ValueCheck(type="int", name="vbase_duo_user", value="8"),
        ]
        self.expect_expr(
            "vbuser", result_type="VBaseDiamondUser *", result_children=vbuser_children
        )

        # Test 'Combined'.
        # struct Combined : MultiWrap1, VBaseDiamondUser, MultiWrap2, Single
        combined = get("combined")
        combined_addr = combined.GetValueAsAddress()
        combined_name = combined.GetTypeName() or ""
        self.assertNotEqual(combined_addr, lldb.LLDB_INVALID_ADDRESS)
        self.assertGreater(len(combined_name), 0)

        combined_mw1 = get("combined_mw1")
        self.assertEqual(combined_addr, combined_mw1.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw1.GetTypeName())
        combined_mw1_multi = get("combined_mw1_multi")
        self.assertEqual(combined_addr, combined_mw1_multi.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw1_multi.GetTypeName())
        combined_mw1_multi_b1 = get("combined_mw1_multi_b1")
        self.assertEqual(combined_addr, combined_mw1_multi_b1.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw1_multi_b1.GetTypeName())
        combined_mw1_multi_b2 = get("combined_mw1_multi_b2")
        self.assertEqual(combined_addr, combined_mw1_multi_b2.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw1_multi_b2.GetTypeName())

        combined_vbuser = get("combined_vbuser")
        self.assertEqual(combined_addr, combined_vbuser.GetValueAsAddress())
        self.assertEqual(combined_name, combined_vbuser.GetTypeName())

        combined_vbuser_vbase = get("combined_vbuser_vbase")
        if is_windows_msvc and not has_rtti:
            vbuser_vbase_addr = combined_vbuser_vbase.GetValueAsAddress()
            self.assertNotEqual(vbuser_vbase_addr, lldb.LLDB_INVALID_ADDRESS)
            self.assertNotEqual(combined_addr, vbuser_vbase_addr)
            self.assertNotEqual(combined_name, combined_vbuser_vbase.GetTypeName())
        else:
            self.assertEqual(combined_addr, combined_vbuser_vbase.GetValueAsAddress())
            self.assertEqual(combined_name, combined_vbuser_vbase.GetTypeName())

        combined_mw2 = get("combined_mw2")
        self.assertEqual(combined_addr, combined_mw2.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw2.GetTypeName())
        combined_mw2_multi = get("combined_mw2_multi")
        self.assertEqual(combined_addr, combined_mw2_multi.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw2_multi.GetTypeName())
        combined_mw2_multi_b1 = get("combined_mw2_multi_b1")
        self.assertEqual(combined_addr, combined_mw2_multi_b1.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw2_multi_b1.GetTypeName())
        combined_mw2_multi_b2 = get("combined_mw2_multi_b2")
        self.assertEqual(combined_addr, combined_mw2_multi_b2.GetValueAsAddress())
        self.assertEqual(combined_name, combined_mw2_multi_b2.GetTypeName())

        combined_single = get("combined_single")
        self.assertEqual(combined_addr, combined_single.GetValueAsAddress())
        self.assertEqual(combined_name, combined_single.GetTypeName())
        combined_single_base1 = get("combined_single_base1")
        self.assertEqual(combined_addr, combined_single_base1.GetValueAsAddress())
        self.assertEqual(combined_name, combined_single_base1.GetTypeName())

        combined_children = [
            ValueCheck(
                type="MultiWrap1",
                children=[
                    ValueCheck(type="Multiple", children=multi_children),
                    ValueCheck(type="int", name="multi_wrap1", value="9"),
                ],
            ),
            ValueCheck(type="VBaseDiamondUser", children=vbuser_children),
            ValueCheck(
                type="MultiWrap2",
                children=[
                    ValueCheck(type="Multiple", children=multi_children),
                    ValueCheck(type="int", name="multi_wrap2", value="10"),
                ],
            ),
            ValueCheck(type="Single", children=single_children),
            ValueCheck(type="int", name="combined", value="11"),
        ]
        self.expect_expr(
            "combined", result_type="Combined *", result_children=combined_children
        )

        # Test 'TwoBase1s'
        # struct TwoBase1s : virtual VBaseWithBase, Single

        tb1 = get("tb1")
        tb1_addr = tb1.GetValueAsAddress()
        tb1_name = tb1.GetTypeName() or ""
        self.assertNotEqual(tb1_addr, lldb.LLDB_INVALID_ADDRESS)
        self.assertGreater(len(tb1_name), 0)

        tb1_vb = get("tb1_vb")
        tb1_vb_b1 = get("tb1_vb_b1")
        if is_windows_msvc and not has_rtti:
            # Anything that goes through the virtual base is unknown to us, so
            # it should not have a dynamic type.
            tb1_vb_addr = tb1_vb.GetValueAsAddress()
            tb1_vb_b1_addr = tb1_vb.GetValueAsAddress()
            self.assertNotEqual(tb1_vb_addr, lldb.LLDB_INVALID_ADDRESS)
            self.assertNotEqual(tb1_vb_b1_addr, lldb.LLDB_INVALID_ADDRESS)
            self.assertNotEqual(tb1_addr, tb1_vb_addr)
            self.assertNotEqual(tb1_addr, tb1_vb_b1_addr)
            self.assertNotEqual(tb1_name, tb1_vb.GetTypeName())
            self.assertNotEqual(tb1_name, tb1_vb_b1.GetTypeName())
        else:
            #
            self.assertEqual(tb1_addr, tb1_vb.GetValueAsAddress())
            self.assertEqual(tb1_name, tb1_vb.GetTypeName())
            self.assertEqual(tb1_addr, tb1_vb_b1.GetValueAsAddress())
            self.assertEqual(tb1_name, tb1_vb_b1.GetTypeName())

        tb1_single = get("tb1_single")
        self.assertEqual(tb1_addr, tb1_single.GetValueAsAddress())
        self.assertEqual(tb1_name, tb1_single.GetTypeName())
        tb1_single_b1 = get("tb1_single_b1")
        self.assertEqual(tb1_addr, tb1_single_b1.GetValueAsAddress())
        self.assertEqual(tb1_name, tb1_single_b1.GetTypeName())

        tb1_children = [
            ValueCheck(
                type="VBaseWithBase",
                children=[
                    ValueCheck(
                        type="Base1",
                        children=[ValueCheck(type="int", name="base1", value="1")],
                    ),
                    ValueCheck(type="int", name="vbase_with_base", value="12"),
                ],
            ),
            ValueCheck(type="Single", children=single_children),
            ValueCheck(type="int", name="two_base1s", value="13"),
        ]
        if not vbases_at_start:
            tb1_children.insert(1, tb1_children.pop(0))
        self.expect_expr("tb1", result_type="TwoBase1s *", result_children=tb1_children)

    def _check_get(self, frame: lldb.SBFrame, name: str) -> lldb.SBValue:
        var = frame.FindVariable(name, lldb.eDynamicCanRunTarget)
        self.assertTrue(var, f"could not find '{name}'")
        return var
