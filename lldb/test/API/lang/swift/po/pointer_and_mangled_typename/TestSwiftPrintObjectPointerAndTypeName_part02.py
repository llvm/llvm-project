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

    @skipEmbeddedSwift
    @swiftTest
    def test_enum(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break enum", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ Enum"])
        self._filecheck("ENUM")
        # CHECK-ENUM: stringForPrintObject(UnsafeRawPointer(bitPattern: {{.*}}), mangledTypeName: "1a4EnumOD")

    @skipEmbeddedSwift
    @swiftTest
    def test_generic_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break generic struct", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ GenericStruct<String>"])
        self._filecheck("GEN-STRUCT")
        # CHECK-GEN-STRUCT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a13GenericStructVySSGD")

    @skipEmbeddedSwift
    @swiftTest
    def test_generic_class(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break generic class", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["<GenericClass<String>: 0x"])
        self._filecheck("GEN-CLASS")
        # CHECK-GEN-CLASS: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a12GenericClassCySSGD")

    @skipEmbeddedSwift
    @swiftTest
    def test_generic_enum(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break generic enum", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ GenericEnum<String>"])
        self._filecheck("GEN-ENUM")
        # CHECK-GEN-ENUM: stringForPrintObject(UnsafeRawPointer(bitPattern: {{.*}}), mangledTypeName: "1a11GenericEnumOySSGD")

