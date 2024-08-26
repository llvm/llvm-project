"""Test the SBModule::FindTypes."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestSBModuleFindTypes(TestBase):
    def test_lookup_in_template_scopes(self):
        self.build()
        spec = lldb.SBModuleSpec()
        spec.SetFileSpec(lldb.SBFileSpec(self.getBuildArtifact()))
        module = lldb.SBModule(spec)

        self.assertEqual(
            set([t.GetName() for t in module.FindTypes("LookMeUp")]),
            set(
                [
                    "ns1::Foo<void>::LookMeUp",
                    "ns2::Bar<void>::LookMeUp",
                    "ns1::Foo<ns2::Bar<void> >::LookMeUp",
                ]
            ),
        )

        self.assertEqual(
            set([t.GetName() for t in module.FindTypes("ns1::Foo<void>::LookMeUp")]),
            set(["ns1::Foo<void>::LookMeUp"]),
        )

        self.assertEqual(
            set(
                [
                    t.GetName()
                    for t in module.FindTypes("ns1::Foo<ns2::Bar<void> >::LookMeUp")
                ]
            ),
            set(["ns1::Foo<ns2::Bar<void> >::LookMeUp"]),
        )
