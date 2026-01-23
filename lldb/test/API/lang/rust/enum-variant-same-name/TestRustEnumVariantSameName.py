"""Test that lldb recognizes enum variant emitted by Rust compiler """
import logging

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from RustEnumValue import RustEnumValue


class TestRustEnumStructs(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "main.yaml")
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)
        self.dbg.CreateTarget(obj_path)

    def getFromGlobal(self, name):
        values = self.target().FindGlobalVariables(name, 1)
        self.assertEqual(values.GetSize(), 1)
        return RustEnumValue(values[0])

    def test_enum_instance(self):
        # static ENUM_INSTANCE: A = A::A(B::B(10));
        value = self.getFromGlobal("ENUM_INSTANCE").getCurrentValue()
        self.assertEqual(value.GetType().GetDisplayTypeName(), "main::A::A")

        value_b = RustEnumValue(value.GetChildAtIndex(0))
        self.assertEqual(
            value_b.getCurrentValue()
            .GetChildAtIndex(0)
            .GetData()
            .GetUnsignedInt8(lldb.SBError(), 0),
            10,
        )
