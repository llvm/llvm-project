"""
Test 'target modules dump symtab -m' doesn't demangle symbols.
"""

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    @no_debug_info_test
    def test(self):
        src_dir = self.getSourceDir()
        yaml_path = os.path.join(src_dir, "a.yaml")
        yaml_base, ext = os.path.splitext(yaml_path)
        obj_path = self.getBuildArtifact("main.o")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)
        self.assertTrue(target, VALID_TARGET)

        # First test that we demangle by default and our mangled symbol isn't in the output.
        self.expect("target modules dump symtab", substrs=["foo::bar(int)"])
        self.expect("target modules dump symtab", matching=False, substrs=["_ZN3foo3barEi"])

        # Turn off demangling and make sure that we now see the mangled name in the output.
        self.expect("target modules dump symtab -m", substrs=["_ZN3foo3barEi"])
