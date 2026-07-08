import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


class TestCase(TestBase):

    def setUp(self):
        TestBase.setUp(self)

        self.log = self.getBuildArtifact("expr.log")
        self.runCmd(f"log enable lldb expr -f {self.log}")

    def _filecheck(self, key):
        self.filecheck_log(self.log, __file__, f"-check-prefix=CHECK-{key}")

    @swiftTest
    @skipEmbeddedSwift
    def test_int(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break int", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["2025"])
        self._filecheck("INT")
        # CHECK-INT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "SiD")

    @swiftTest
    @skipEmbeddedSwift
    def test_string(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break string", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["Po"])
        self._filecheck("STRING")
        # CHECK-STRING: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "SSD")

    @skipEmbeddedSwift
    @swiftTest
    def test_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break struct", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ Struct"])
        self._filecheck("STRUCT")
        # CHECK-STRUCT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a6StructVD")

    @skipEmbeddedSwift
    @swiftTest
    def test_class(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break class", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["<Class: 0x"])
        self._filecheck("CLASS")
        # CHECK-CLASS: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a5ClassCD")

