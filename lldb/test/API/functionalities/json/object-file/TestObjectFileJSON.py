import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import json
import uuid
import os
import shutil
import time


class TestObjectFileJSON(TestBase):
    TRIPLE = "arm64-apple-macosx13.0.0"

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    def emitJSON(self, data, path):
        json_object = json.dumps(data, indent=4)
        with open(path, "w") as outfile:
            outfile.write(json_object)

    def toModuleSpec(self, path):
        module_spec = lldb.SBModuleSpec()
        module_spec.SetFileSpec(lldb.SBFileSpec(path))
        return module_spec

    @no_debug_info_test
    def test_target(self):
        triple = "arm64-apple-macosx13.0.0"
        data = {
            "triple": triple,
            "uuid": str(uuid.uuid4()),
            "type": "executable",
        }

        json_object_file = self.getBuildArtifact("a.json")
        self.emitJSON(data, json_object_file)

        target = self.dbg.CreateTarget(json_object_file)
        self.assertTrue(target.IsValid())
        self.assertEqual(target.GetTriple(), triple)

    @no_debug_info_test
    def test_module(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)

        data = {
            "triple": target.GetTriple(),
            "uuid": str(uuid.uuid4()),
        }

        json_object_file_b = self.getBuildArtifact("b.json")
        self.emitJSON(data, json_object_file_b)

        module = target.AddModule(self.toModuleSpec(json_object_file_b))
        self.assertFalse(module.IsValid())
        TEXT_file_addr = 0x100000000
        DATA_file_addr = 0x100001000
        foo_file_addr = TEXT_file_addr + 0x100
        bar_file_addr = DATA_file_addr + 0x10
        TEXT_size = 0x222
        DATA_size = 0x333
        foo_size = 0x11
        bar_size = 0x22
        slide = 0x100000000
        data = {
            "triple": target.GetTriple(),
            "uuid": str(uuid.uuid4()),
            "type": "sharedlibrary",
            "sections": [
                {
                    "name": "__TEXT",
                    "type": "container",
                    "address": TEXT_file_addr,
                    "size": TEXT_size,
                    "read": True,
                    "write": False,
                    "execute": True,
                    "subsections": [
                        {
                            "name": "__text",
                            "type": "code",
                            "address": TEXT_file_addr,
                            "size": TEXT_size,
                            "read": True,
                            "write": False,
                            "execute": True,
                        }
                    ],
                },
                {
                    "name": "__DATA",
                    "type": "data",
                    "address": DATA_file_addr,
                    "size": DATA_size,
                    "read": True,
                    "write": True,
                    "execute": False,
                },
            ],
            "symbols": [
                {
                    "name": "foo",
                    "type": "code",
                    "address": foo_file_addr,
                    "size": foo_size,
                },
                {
                    "name": "bar",
                    "type": "data",
                    "address": bar_file_addr,
                    "size": bar_size,
                },
            ],
        }

        json_object_file_c = self.getBuildArtifact("c.json")
        self.emitJSON(data, json_object_file_c)

        module = target.AddModule(self.toModuleSpec(json_object_file_c))
        self.assertTrue(module.IsValid())

        TEXT_section = module.GetSectionAtIndex(0)
        self.assertTrue(TEXT_section.IsValid())
        self.assertEqual(TEXT_section.GetName(), "__TEXT")
        self.assertEqual(TEXT_section.file_addr, TEXT_file_addr)
        self.assertEqual(TEXT_section.size, TEXT_size)
        self.assertEqual(TEXT_section.GetSectionType(), lldb.eSectionTypeContainer)
        self.assertEqual(TEXT_section.GetNumSubSections(), 1)
        text_permissions = TEXT_section.GetPermissions()
        self.assertTrue((text_permissions & lldb.ePermissionsReadable) != 0)
        self.assertFalse((text_permissions & lldb.ePermissionsWritable) != 0)
        self.assertTrue((text_permissions & lldb.ePermissionsExecutable) != 0)

        text_section = TEXT_section.GetSubSectionAtIndex(0)
        self.assertTrue(text_section.IsValid())
        self.assertEqual(text_section.GetName(), "__text")
        self.assertEqual(text_section.file_addr, TEXT_file_addr)
        self.assertEqual(text_section.size, TEXT_size)
        self.assertEqual(text_section.GetSectionType(), lldb.eSectionTypeCode)
        self.assertEqual(text_section.GetNumSubSections(), 0)
        text_permissions = text_section.GetPermissions()
        self.assertTrue((text_permissions & lldb.ePermissionsReadable) != 0)
        self.assertFalse((text_permissions & lldb.ePermissionsWritable) != 0)
        self.assertTrue((text_permissions & lldb.ePermissionsExecutable) != 0)

        DATA_section = module.GetSectionAtIndex(1)
        self.assertTrue(DATA_section.IsValid())
        self.assertEqual(DATA_section.GetName(), "__DATA")
        self.assertEqual(DATA_section.file_addr, DATA_file_addr)
        self.assertEqual(DATA_section.size, DATA_size)
        self.assertEqual(DATA_section.GetSectionType(), lldb.eSectionTypeData)
        data_permissions = DATA_section.GetPermissions()
        self.assertTrue((data_permissions & lldb.ePermissionsReadable) != 0)
        self.assertTrue((data_permissions & lldb.ePermissionsWritable) != 0)
        self.assertFalse((data_permissions & lldb.ePermissionsExecutable) != 0)

        foo_symbol = module.FindSymbol("foo")
        self.assertTrue(foo_symbol.IsValid())
        self.assertEqual(foo_symbol.addr.GetFileAddress(), foo_file_addr)
        self.assertEqual(foo_symbol.GetSize(), foo_size)

        bar_symbol = module.FindSymbol("bar")
        self.assertTrue(bar_symbol.IsValid())
        self.assertEqual(bar_symbol.addr.GetFileAddress(), bar_file_addr)
        self.assertEqual(bar_symbol.GetSize(), bar_size)

        error = target.SetSectionLoadAddress(TEXT_section, TEXT_file_addr + slide)
        self.assertSuccess(error)
        error = target.SetSectionLoadAddress(DATA_section, DATA_file_addr + slide)
        self.assertSuccess(error)
        self.assertEqual(foo_symbol.addr.GetLoadAddress(target), foo_file_addr + slide)
        self.assertEqual(bar_symbol.addr.GetLoadAddress(target), bar_file_addr + slide)
