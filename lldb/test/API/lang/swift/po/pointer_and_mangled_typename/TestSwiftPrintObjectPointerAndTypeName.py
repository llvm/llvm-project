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
    @expectedFailureWindows
    def test_int(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break int", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["2025"])
        self._filecheck("INT")
        # CHECK-INT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "SiD")

    @swiftTest
    @expectedFailureWindows
    def test_string(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break string", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["Po"])
        self._filecheck("STRING")
        # CHECK-STRING: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "SSD")

    @swiftTest
    @expectedFailureWindows
    def test_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break struct", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ Struct"])
        self._filecheck("STRUCT")
        # CHECK-STRUCT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a6StructVD")

    @swiftTest
    @expectedFailureWindows
    def test_class(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break class", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["<Class: 0x"])
        self._filecheck("CLASS")
        # CHECK-CLASS: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a5ClassCD")

    @swiftTest
    @expectedFailureWindows
    def test_enum(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break enum", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ Enum"])
        self._filecheck("ENUM")
        # CHECK-ENUM: stringForPrintObject(UnsafeRawPointer(bitPattern: {{.*}}), mangledTypeName: "1a4EnumOD")

    @swiftTest
    @expectedFailureWindows
    def test_generic_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break generic struct", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ GenericStruct<String>"])
        self._filecheck("GEN-STRUCT")
        # CHECK-GEN-STRUCT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a13GenericStructVySSGD")

    @swiftTest
    @expectedFailureWindows
    def test_generic_class(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break generic class", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["<GenericClass<String>: 0x"])
        self._filecheck("GEN-CLASS")
        # CHECK-GEN-CLASS: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a12GenericClassCySSGD")

    @swiftTest
    @expectedFailureWindows
    def test_generic_enum(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break generic enum", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["▿ GenericEnum<String>"])
        self._filecheck("GEN-ENUM")
        # CHECK-GEN-ENUM: stringForPrintObject(UnsafeRawPointer(bitPattern: {{.*}}), mangledTypeName: "1a11GenericEnumOySSGD")

    @swiftTest
    @expectedFailureWindows
    def test_described_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break described struct", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedStruct"])
        self._filecheck("DESC-STRUCT")
        # CHECK-DESC-STRUCT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a15DescribedStructVD")

    @swiftTest
    @expectedFailureWindows
    def test_described_class(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break described class", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedClass"])
        self._filecheck("DESC-CLASS")
        # CHECK-DESC-CLASS: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a14DescribedClassCD")

    @swiftTest
    @expectedFailureWindows
    def test_described_enum(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break described enum", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedEnum"])
        self._filecheck("DESC-ENUM")
        # CHECK-DESC-ENUM: stringForPrintObject(UnsafeRawPointer(bitPattern: {{.*}}), mangledTypeName: "1a13DescribedEnumOD")
