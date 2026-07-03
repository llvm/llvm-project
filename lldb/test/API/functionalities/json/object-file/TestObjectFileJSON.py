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
        text_size = 0x20
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
                    "user_id": 0x100,
                    "name": "__PAGEZERO",
                    "type": "container",
                    "address": 0,
                    "size": 0x100000000,
                    "flags": 0x101
              },
              {
                    "user_id": 0x200,
                    "name": "__TEXT",
                    "type": "container",
                    "address": TEXT_file_addr,
                    "size": TEXT_size,
                    "flags": 0x202,
                    "file_offset": 0,
                    "file_size": TEXT_size,
                    "read": True,
                    "write": False,
                    "execute": True,
                    "subsections": [
                        {
                            "name": "__text",
                            "type": "code",
                            "address": TEXT_file_addr,
                            "size": text_size,
                            "alignment": 2,
                            "read": True,
                            "write": False,
                            "execute": True,
                        },
                        {
                            "name": "__fake",
                            "address": TEXT_file_addr + 1 * text_size,
                            "size": text_size,
                            "fake": True
                        },
                        {
                            "name": "__encrypted",
                            "address": TEXT_file_addr + 2 * text_size,
                            "size": text_size,
                            "encrypted": True
                        },
                        {
                            "name": "__tls",
                            "address": TEXT_file_addr + 2 * text_size,
                            "size": text_size,
                            "thread_specific": True
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
                    "flags": 0x303,
                    "file_offset": DATA_file_addr - TEXT_file_addr,
                    "file_size": DATA_size,
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
        self.assertEqual(TEXT_section.GetName(), "__PAGEZERO")
        self.assertEqual(TEXT_section.file_addr, 0)
        self.assertEqual(TEXT_section.size, 0x100000000)
        self.assertEqual(TEXT_section.GetSectionType(), lldb.eSectionTypeContainer)
        self.assertEqual(TEXT_section.GetNumSubSections(), 0)
        text_permissions = TEXT_section.GetPermissions()
        self.assertFalse((text_permissions & lldb.ePermissionsReadable) != 0)
        self.assertFalse((text_permissions & lldb.ePermissionsWritable) != 0)
        self.assertFalse((text_permissions & lldb.ePermissionsExecutable) != 0)

        TEXT_section = module.GetSectionAtIndex(1)
        self.assertTrue(TEXT_section.IsValid())
        self.assertEqual(TEXT_section.GetName(), "__TEXT")
        self.assertEqual(TEXT_section.file_addr, TEXT_file_addr)
        self.assertEqual(TEXT_section.size, TEXT_size)
        self.assertEqual(TEXT_section.file_offset, 0)
        self.assertEqual(TEXT_section.file_size, TEXT_size)
        self.assertEqual(TEXT_section.GetSectionType(), lldb.eSectionTypeContainer)
        self.assertEqual(TEXT_section.GetNumSubSections(), 4)
        text_permissions = TEXT_section.GetPermissions()
        self.assertTrue((text_permissions & lldb.ePermissionsReadable) != 0)
        self.assertFalse((text_permissions & lldb.ePermissionsWritable) != 0)
        self.assertTrue((text_permissions & lldb.ePermissionsExecutable) != 0)

        text_section = TEXT_section.GetSubSectionAtIndex(0)
        self.assertEqual(text_section.GetName(), "__text")
        self.assertEqual(text_section.size, text_size)
        self.assertEqual(text_section.GetAlignment(), 4)
        self.assertEqual(text_section.GetSectionType(), lldb.eSectionTypeCode)
        self.assertEqual(text_section.GetNumSubSections(), 0)
        text_permissions = text_section.GetPermissions()
        self.assertTrue((text_permissions & lldb.ePermissionsReadable) != 0)
        self.assertFalse((text_permissions & lldb.ePermissionsWritable) != 0)
        self.assertTrue((text_permissions & lldb.ePermissionsExecutable) != 0)

        DATA_section = module.GetSectionAtIndex(2)
        self.assertTrue(DATA_section.IsValid())
        self.assertEqual(DATA_section.GetName(), "__DATA")
        self.assertEqual(DATA_section.file_addr, DATA_file_addr)
        self.assertEqual(DATA_section.size, DATA_size)
        self.assertEqual(DATA_section.file_offset, DATA_file_addr - TEXT_file_addr)
        self.assertEqual(DATA_section.file_size, DATA_size)
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

        # Verify the user_ids and flags are set correctly since there is no API
        # for this on lldb.SBSection
        self.expect("target modules dump sections c.json",
            substrs = [
                "0x0000000000000100 container              [0x0000000000000000-0x0000000100000000)  ---  0x00000000 0x00000000 0x00000101 c.json.__PAGEZERO",
                "0x0000000000000200 container              [0x0000000100000000-0x0000000100000222)  r-x  0x00000000 0x00000222 0x00000202 c.json.__TEXT",
                "0x0000000000000001 code                   [0x0000000100000000-0x0000000100000020)  r-x  0x00000000 0x00000000 0x00000000 c.json.__TEXT.__text",
                "0x0000000000000002 code                   [0x0000000100000020-0x0000000100000040)  ---  0x00000000 0x00000000 0x00000000 c.json.__TEXT.__fake",
                "0x0000000000000003 code                   [0x0000000100000040-0x0000000100000060)  ---  0x00000000 0x00000000 0x00000000 c.json.__TEXT.__encrypted",
                "0x0000000000000004 code                   [0x0000000100000040-0x0000000100000060)  ---  0x00000000 0x00000000 0x00000000 c.json.__TEXT.__tls",
                "0x0000000000000005 data                   [0x0000000100001000-0x0000000100001333)  rw-  0x00001000 0x00000333 0x00000303 c.json.__DATA"
            ])

        error = target.SetSectionLoadAddress(TEXT_section, TEXT_file_addr + slide)
        self.assertSuccess(error)
        error = target.SetSectionLoadAddress(DATA_section, DATA_file_addr + slide)
        self.assertSuccess(error)
        self.assertEqual(foo_symbol.addr.GetLoadAddress(target), foo_file_addr + slide)
        self.assertEqual(bar_symbol.addr.GetLoadAddress(target), bar_file_addr + slide)
