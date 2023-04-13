import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import json
import uuid
import os
import shutil


class TestObjectFileJSON(TestBase):
    TRIPLE = "arm64-apple-macosx13.0.0"

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    def emitJSON(self, data, path):
        json_object = json.dumps(data, indent=4)
        json_object_file = self.getBuildArtifact("a.json")
        with open(json_object_file, "w") as outfile:
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

        json_object_file = self.getBuildArtifact("a.json")
        self.emitJSON(data, json_object_file)

        module = target.AddModule(self.toModuleSpec(json_object_file))
        self.assertFalse(module.IsValid())

        data = {
            "triple": target.GetTriple(),
            "uuid": str(uuid.uuid4()),
            "type": "sharedlibrary",
            "sections": [
                {
                    "name": "__TEXT",
                    "type": "code",
                    "address": 0,
                    "size": 0x222,
                }
            ],
            "symbols": [
                {
                    "name": "foo",
                    "address": 0x100,
                    "size": 0x11,
                }
            ],
        }
        self.emitJSON(data, json_object_file)

        module = target.AddModule(self.toModuleSpec(json_object_file))
        self.assertTrue(module.IsValid())

        section = module.GetSectionAtIndex(0)
        self.assertTrue(section.IsValid())
        self.assertEqual(section.GetName(), "__TEXT")
        self.assertEqual(section.file_addr, 0x0)
        self.assertEqual(section.size, 0x222)

        symbol = module.FindSymbol("foo")
        self.assertTrue(symbol.IsValid())
        self.assertEqual(symbol.addr.GetFileAddress(), 0x100)
        self.assertEqual(symbol.GetSize(), 0x11)

        error = target.SetSectionLoadAddress(section, 0x1000)
        self.assertSuccess(error)
        self.assertEqual(symbol.addr.GetLoadAddress(target), 0x1100)
