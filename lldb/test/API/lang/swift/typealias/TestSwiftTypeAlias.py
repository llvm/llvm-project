from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftTypeAlias(TestBase):
    @swiftTest
    def test(self):
        """Test type aliases are only searched in the debug info once"""
        self.build()
        log = self.getBuildArtifact("dwarf.log")
        self.expect("log enable dwarf lookups -f " + log)

        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift"), extra_images=['Dylib'])
        self.expect("target variable foo", substrs=["(Dylib.MyAlias)", "23"])
        self.expect("target variable bar",
                    substrs=["(Dylib.MyGenericAlias<Dylib.MyAlias>)", "42"])
