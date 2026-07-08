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
    def test_described_struct(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break described struct", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedStruct"])
        self._filecheck("DESC-STRUCT")
        # CHECK-DESC-STRUCT: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a15DescribedStructVD")

    @swiftTest
    @skipEmbeddedSwift
    def test_described_class(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break described class", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedClass"])
        self._filecheck("DESC-CLASS")
        # CHECK-DESC-CLASS: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a14DescribedClassCD")

    @swiftTest
    @skipEmbeddedSwift
    def test_described_enum(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break described enum", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedEnum"])
        self._filecheck("DESC-ENUM")
        # CHECK-DESC-ENUM: stringForPrintObject(UnsafeRawPointer(bitPattern: {{.*}}), mangledTypeName: "1a13DescribedEnumOD")

    @swiftTest
    @skipEmbeddedSwift
    def test_class_only_protocol(self):
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "break class-only protocol", lldb.SBFileSpec("main.swift")
        )
        self.expect("po value", substrs=["DescribedConformance"])
        self._filecheck("CLASS-ONLY-PROTOCOL")
        # CHECK-CLASS-ONLY-PROTOCOL: stringForPrintObject(UnsafeRawPointer(bitPattern: {{[0-9]+}}), mangledTypeName: "1a20DescribedConformanceCD")
