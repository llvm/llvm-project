"""
Test breakpoint on function with abi_tags.
"""

import lldb
from typing import List, Set, TypedDict
from lldbsuite.test.decorators import skipIfWindows
from lldbsuite.test.lldbtest import VALID_TARGET, TestBase


class Case(TypedDict, total=True):
    name: str
    matches: Set[str]


@skipIfWindows  # abi_tags is not supported
class TestCPPBreakpointLocationsAbiTag(TestBase):

    def verify_breakpoint_names(self, target: lldb.SBTarget, bp_dict: Case):
        name = bp_dict["name"]
        matches = bp_dict["matches"]
        bp: lldb.SBBreakpoint = target.BreakpointCreateByName(name)

        for location in bp:
            self.assertTrue(location.IsValid(), f"Expected valid location {location}")

        expected_matches = set(location.addr.function.name for location in bp)

        self.assertSetEqual(expected_matches, matches)

    def test_breakpoint_name_with_abi_tag(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target: lldb.SBTarget = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        test_cases: List[Case] = [
            Case(
                name="foo",
                matches={
                    "foo[abi:FOO]()",
                    "StaticStruct[abi:STATIC_STRUCT]::foo[abi:FOO][abi:FOO2]()",
                    "Struct[abi:STRUCT]::foo[abi:FOO]()",
                    "ns::NamespaceStruct[abi:NAMESPACE_STRUCT]::foo[abi:FOO]()",
                    "ns::foo[abi:NAMESPACE_FOO]()",
                    "TemplateStruct[abi:TEMPLATE_STRUCT]<int>::foo[abi:FOO]()",
                    "void TemplateStruct[abi:TEMPLATE_STRUCT]<int>::foo[abi:FOO_TEMPLATE]<long>(long)",
                },
            ),
            Case(
                name="StaticStruct::foo",
                matches={"StaticStruct[abi:STATIC_STRUCT]::foo[abi:FOO][abi:FOO2]()"},
            ),
            Case(name="Struct::foo", matches={"Struct[abi:STRUCT]::foo[abi:FOO]()"}),
            Case(
                name="TemplateStruct::foo",
                matches={
                    "TemplateStruct[abi:TEMPLATE_STRUCT]<int>::foo[abi:FOO]()",
                    "void TemplateStruct[abi:TEMPLATE_STRUCT]<int>::foo[abi:FOO_TEMPLATE]<long>(long)",
                },
            ),
            Case(name="ns::foo", matches={"ns::foo[abi:NAMESPACE_FOO]()"}),
            # operators
            Case(
                name="operator<",
                matches={
                    "Struct[abi:STRUCT]::operator<(int)",
                    "bool TemplateStruct[abi:TEMPLATE_STRUCT]<int>::operator<[abi:OPERATOR]<int>(int)",
                },
            ),
            Case(
                name="TemplateStruct::operator<<",
                matches={
                    "bool TemplateStruct[abi:TEMPLATE_STRUCT]<int>::operator<<[abi:operator]<int>(int)"
                },
            ),
            Case(
                name="operator<<",
                matches={
                    "bool TemplateStruct[abi:TEMPLATE_STRUCT]<int>::operator<<[abi:operator]<int>(int)"
                },
            ),
            Case(
                name="operator==",
                matches={"operator==[abi:OPERATOR](wrap_int const&, wrap_int const&)"},
            ),
        ]

        for case in test_cases:
            self.verify_breakpoint_names(target, case)
