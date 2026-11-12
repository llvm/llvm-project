"""
Test some SBModuleSpec APIs.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class ModuleSpecAPIsTestCase(TestBase):
    def test_object_offset_and_size(self):
        module_spec = lldb.SBModuleSpec()
        self.assertEqual(module_spec.GetObjectOffset(), 0)
        self.assertEqual(module_spec.GetObjectSize(), 0)

        module_spec.SetObjectOffset(4096)
        self.assertEqual(module_spec.GetObjectOffset(), 4096)

        module_spec.SetObjectSize(3600)
        self.assertEqual(module_spec.GetObjectSize(), 3600)

        module_spec.Clear()
        self.assertEqual(module_spec.GetObjectOffset(), 0)
        self.assertEqual(module_spec.GetObjectSize(), 0)
