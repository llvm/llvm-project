import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

# This test checks that source code location is shown correctly
# when DWARF5 debug information is used.


class TestTargetSourceMap(TestBase):
    @skipIf(
        archs="aarch64",
        oslist="linux",
        bugnumber="https://bugs.llvm.org/show_bug.cgi?id=44180",
    )
    def test_source_map(self):
        # Set the target soure map to map "./" to the current test directory.
        yaml_path = os.path.join(self.getSourceDir(), "a.yaml")
        obj_path = self.getBuildArtifact("a.out")
        self.yaml2obj(yaml_path, obj_path)

        # Create a target with the object file we just created from YAML
        target = self.dbg.CreateTarget(obj_path)

        # Check we are able to show the locations properly.
        self.expect(
            "b main",
            VALID_BREAKPOINT_LOCATION,
            substrs=["main + 13 at test.cpp:2:3, address = 0x000000000040052d"],
        )

        self.expect(
            "b foo",
            VALID_BREAKPOINT_LOCATION,
            substrs=["foo() + 4 at test.cpp:6:1, address = 0x0000000000400534"],
        )
