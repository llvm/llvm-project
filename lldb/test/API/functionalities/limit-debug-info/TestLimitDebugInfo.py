"""
Test completing types using information from other shared libraries.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LimitDebugInfoTestCase(TestBase):

    def _check_type(self, target, name):
        exe = target.FindModule(lldb.SBFileSpec("a.out"))
        type_ = exe.FindFirstType(name)
        self.trace("type_: %s"%type_)
        self.assertTrue(type_)
        self.assertTrue(type_.IsTypeComplete())
        base = type_.GetDirectBaseClassAtIndex(0).GetType()
        self.trace("base:%s"%base)
        self.assertTrue(base)
        self.assertEquals(base.GetNumberOfFields(), 0)
        self.assertFalse(base.IsTypeComplete())

    def _check_debug_info_is_limited(self, target):
        # Without other shared libraries we should only see the member declared
        # in the derived class. This serves as a sanity check that we are truly
        # building with limited debug info.
        self._check_type(target, "InheritsFromOne")
        self._check_type(target, "InheritsFromTwo")

        # Check that the statistics show that we had incomplete debug info.
        stats = self.get_stats()
        # Find the a.out module info in the stats and verify it has the
        # "debugInfoHadIncompleteTypes" key value pair set to True
        exe_module_found = False
        for module in stats['modules']:
            if module['path'].endswith('a.out'):
                self.assertTrue(module['debugInfoHadIncompleteTypes'])
                exe_module_found = True
                break
        self.assertTrue(exe_module_found)
        # Verify that "totalModuleCountWithIncompleteTypes" at the top level
        # is greater than zero which shows we had incomplete debug info in a
        # module
        self.assertGreater(stats['totalModuleCountWithIncompleteTypes'], 0)


    def _check_incomplete_frame_variable_output(self):
        # Check that the display of the "frame variable" output identifies the
        # incomplete types. Currently the expression parser will find the real
        # definition for a type when running an expression for any forcefully
        # completed types, but "frame variable" won't. I hope to fix this with
        # a follow up patch, but if we don't find the actual definition we
        # should clearly show this to the user by showing which types were
        # incomplete. So this will test verifies the expected output for such
        # types. We also need to verify the standard "frame variable" output
        # which will inline all of the members on one line, versus the full
        # output from "frame variable --raw" and a few other options.
        # self.expect("frame variable two_as_member", error=True,
        #     substrs=["no member named 'one' in 'InheritsFromOne'"])

        command_expect_pairs = [
            # Test standard "frame variable" output for types to make sure
            # "<incomplete type>" shows up where we expect it to
            ["var two_as_member", [
                "(TwoAsMember) ::two_as_member = (two = <incomplete type>, member = 47)"]
            ],
            ["var inherits_from_one", [
                "(InheritsFromOne) ::inherits_from_one = (One = <incomplete type>, member = 47)"]
            ],
            ["var inherits_from_two", [
                "(InheritsFromTwo) ::inherits_from_two = (Two = <incomplete type>, member = 47)"]
            ],
            ["var one_as_member", [
                "(OneAsMember) ::one_as_member = (one = <incomplete type>, member = 47)"]
            ],
            ["var two_as_member", [
                "(TwoAsMember) ::two_as_member = (two = <incomplete type>, member = 47)"]
            ],
            ["var array_of_one", [
                "(array::One[3]) ::array_of_one = ([0] = <incomplete type>, [1] = <incomplete type>, [2] = <incomplete type>)"]
            ],
            ["var array_of_two", [
                "(array::Two[3]) ::array_of_two = ([0] = <incomplete type>, [1] = <incomplete type>, [2] = <incomplete type>)"]
            ],
            ["var shadowed_one", [
                "(ShadowedOne) ::shadowed_one = (func_shadow::One = <incomplete type>, member = 47)"]
            ],

            # Now test "frame variable --show-types output" which has multi-line
            # output and should not always show classes that were forcefully
            # completed to the user to let them know they have a type that should
            # have been complete but wasn't.
            ["var --show-types inherits_from_one", [
                "(InheritsFromOne) ::inherits_from_one = {",
                "  (One) One = <incomplete type> {}",
                "  (int) member = 47",
                "}"]
            ],
            ["var --show-types inherits_from_two", [
                "(InheritsFromTwo) ::inherits_from_two = {",
                "  (Two) Two = <incomplete type> {}",
                "  (int) member = 47",
                "}"]
            ],
            ["var  --show-types one_as_member", [
                "(OneAsMember) ::one_as_member = {",
                "  (member::One) one = <incomplete type> {}",
                "  (int) member = 47",
                "}"]
            ],
            ["var  --show-types two_as_member", [
                "(TwoAsMember) ::two_as_member = {",
                "  (member::Two) two = <incomplete type> {}",
                "  (int) member = 47",
                "}"]
            ],
            ["var  --show-types array_of_one", [
                "(array::One[3]) ::array_of_one = {",
                "  (array::One) [0] = <incomplete type> {}",
                "  (array::One) [1] = <incomplete type> {}",
                "  (array::One) [2] = <incomplete type> {}",
                "}"]
            ],
            ["var  --show-types array_of_two", [
                "(array::Two[3]) ::array_of_two = {",
                "  (array::Two) [0] = <incomplete type> {}",
                "  (array::Two) [1] = <incomplete type> {}",
                "  (array::Two) [2] = <incomplete type> {}",
                "}"]
            ],
            ["var  --show-types shadowed_one", [
                "(ShadowedOne) ::shadowed_one = {",
                "  (func_shadow::One) func_shadow::One = <incomplete type> {}",
                "  (int) member = 47",
                "}"]
            ],
        ]
        for command, expect_items in command_expect_pairs:
            self.expect(command, substrs=expect_items)

    @skipIf(bugnumber="pr46284", debug_info="gmodules")
    @skipIfWindows # Clang emits type info even with -flimit-debug-info
    # Requires DW_CC_pass_by_* attributes from Clang 7 to correctly call
    # by-value functions.
    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def test_one_and_two_debug(self):
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self._check_debug_info_is_limited(target)

        lldbutil.run_to_name_breakpoint(self, "main",
                                        extra_images=["one", "two"])

        # But when other shared libraries are loaded, we should be able to see
        # all members.
        self.expect_expr("inherits_from_one.member", result_value="47")
        self.expect_expr("inherits_from_one.one", result_value="142")
        self.expect_expr("inherits_from_two.member", result_value="47")
        self.expect_expr("inherits_from_two.one", result_value="142")
        self.expect_expr("inherits_from_two.two", result_value="242")

        self.expect_expr("one_as_member.member", result_value="47")
        self.expect_expr("one_as_member.one.member", result_value="147")
        self.expect_expr("two_as_member.member", result_value="47")
        self.expect_expr("two_as_member.two.one.member", result_value="147")
        self.expect_expr("two_as_member.two.member", result_value="247")

        self.expect_expr("array_of_one[2].member", result_value="174")
        self.expect_expr("array_of_two[2].one[2].member", result_value="174")
        self.expect_expr("array_of_two[2].member", result_value="274")

        self.expect_expr("get_one().member", result_value="124")
        self.expect_expr("get_two().one().member", result_value="124")
        self.expect_expr("get_two().member", result_value="224")

        self.expect_expr("shadowed_one.member", result_value="47")
        self.expect_expr("shadowed_one.one", result_value="142")

        self._check_incomplete_frame_variable_output()

    @skipIf(bugnumber="pr46284", debug_info="gmodules")
    @skipIfWindows # Clang emits type info even with -flimit-debug-info
    # Requires DW_CC_pass_by_* attributes from Clang 7 to correctly call
    # by-value functions.
    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def test_two_debug(self):
        self.build(dictionary=dict(STRIP_ONE="1"))
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self._check_debug_info_is_limited(target)

        lldbutil.run_to_name_breakpoint(self, "main",
                extra_images=["one", "two"])

        # This time, we should only see the members from the second library.
        self.expect_expr("inherits_from_one.member", result_value="47")
        self.expect("expr inherits_from_one.one", error=True,
            substrs=["no member named 'one' in 'InheritsFromOne'"])
        self.expect_expr("inherits_from_two.member", result_value="47")
        self.expect("expr inherits_from_two.one", error=True,
            substrs=["no member named 'one' in 'InheritsFromTwo'"])
        self.expect_expr("inherits_from_two.two", result_value="242")

        self.expect_expr("one_as_member.member", result_value="47")
        self.expect("expr one_as_member.one.member", error=True,
                substrs=["no member named 'member' in 'member::One'"])
        self.expect_expr("two_as_member.member", result_value="47")
        self.expect("expr two_as_member.two.one.member", error=True,
                substrs=["no member named 'member' in 'member::One'"])
        self.expect_expr("two_as_member.two.member", result_value="247")

        self.expect("expr array_of_one[2].member", error=True,
                substrs=["no member named 'member' in 'array::One'"])
        self.expect("expr array_of_two[2].one[2].member", error=True,
                substrs=["no member named 'member' in 'array::One'"])
        self.expect_expr("array_of_two[2].member", result_value="274")

        self.expect("expr get_one().member", error=True,
                substrs=["calling 'get_one' with incomplete return type 'result::One'"])
        self.expect("expr get_two().one().member", error=True,
                substrs=["calling 'one' with incomplete return type 'result::One'"])
        self.expect_expr("get_two().member", result_value="224")

        self._check_incomplete_frame_variable_output()

    @skipIf(bugnumber="pr46284", debug_info="gmodules")
    @skipIfWindows # Clang emits type info even with -flimit-debug-info
    # Requires DW_CC_pass_by_* attributes from Clang 7 to correctly call
    # by-value functions.
    @skipIf(compiler="clang", compiler_version=['<', '7.0'])
    def test_one_debug(self):
        self.build(dictionary=dict(STRIP_TWO="1"))
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))

        self._check_debug_info_is_limited(target)

        lldbutil.run_to_name_breakpoint(self, "main",
                extra_images=["one", "two"])

        # In this case we should only see the members from the second library.
        # Note that we cannot see inherits_from_two.one because without debug
        # info for "Two", we cannot determine that it in fact inherits from
        # "One".
        self.expect_expr("inherits_from_one.member", result_value="47")
        self.expect_expr("inherits_from_one.one", result_value="142")
        self.expect_expr("inherits_from_two.member", result_value="47")
        self.expect("expr inherits_from_two.one", error=True,
            substrs=["no member named 'one' in 'InheritsFromTwo'"])
        self.expect("expr inherits_from_two.two", error=True,
            substrs=["no member named 'two' in 'InheritsFromTwo'"])

        self.expect_expr("one_as_member.member", result_value="47")
        self.expect_expr("one_as_member.one.member", result_value="147")
        self.expect_expr("two_as_member.member", result_value="47")
        self.expect("expr two_as_member.two.one.member", error=True,
                substrs=["no member named 'one' in 'member::Two'"])
        self.expect("expr two_as_member.two.member", error=True,
                substrs=["no member named 'member' in 'member::Two'"])

        self.expect_expr("array_of_one[2].member", result_value="174")
        self.expect("expr array_of_two[2].one[2].member", error=True,
                substrs=["no member named 'one' in 'array::Two'"])
        self.expect("expr array_of_two[2].member", error=True,
                substrs=["no member named 'member' in 'array::Two'"])

        self.expect_expr("get_one().member", result_value="124")
        self.expect("expr get_two().one().member", error=True,
                substrs=["calling 'get_two' with incomplete return type 'result::Two'"])
        self.expect("expr get_two().member", error=True,
                substrs=["calling 'get_two' with incomplete return type 'result::Two'"])

        self._check_incomplete_frame_variable_output()
