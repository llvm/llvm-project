# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import importlib.util
import pathlib
import subprocess
import sys
import unittest
root = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hub", root/"bridges"/"hub.py")
hub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub)
opt = sys.argv.pop(1) if len(sys.argv)>1 else None

class BridgeTests(unittest.TestCase):
    def test_languages(self):
        records = [hub.reference(l,"sample",b"sample","f","function") for l in sorted(hub.LANGS)]
        self.assertEqual(len({r["reference_id"] for r in records}),6)
        self.assertEqual(records[0],hub.reference(records[0]["language"],"sample",b"sample","f","function"))
        self.assertNotEqual(records[0]["reference_id"],hub.reference(records[0]["language"],"sample",b"changed","f","function")["reference_id"])
        if opt:
            p = subprocess.run([opt],input=hub.emit(records),text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            p = subprocess.run([opt],input=hub.emit(records).replace('schema_version = "0.2"','schema_version = "bad"'),text=True,capture_output=True)
            self.assertNotEqual(p.returncode,0)
    def test_ast(self):
        records=hub.python_records("a.py","def f():\n    raise Exception()\nclass Việt:\n    pass\n".encode())
        self.assertEqual([r["semantic_kind"] for r in records],["function","type"])
        self.assertIn("\\E1",hub.emit(records))
    def test_invalid(self):
        with self.assertRaises(ValueError):
            hub.reference("unknown","a",b"a","f","function")
        with self.assertRaises(ValueError):
            hub.reference("python","a",b"a","","function")
    def test_escaping(self):
        self.assertEqual(hub.mlir_string('a"\\'),'"a\\22\\5C"')

if __name__=="__main__":
    unittest.main()
