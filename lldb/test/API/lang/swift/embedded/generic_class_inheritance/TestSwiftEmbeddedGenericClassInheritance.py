import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedGenericClassInheritance(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test_generic_derived(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )

        self.expect(
            "frame variable derived",
            substrs=[
                "GenericDerived<String, Int>",
                'baseValue = "hello"',
                "derivedValue = 100",
            ],
        )

        self.expect(
            "frame variable triple",
            substrs=[
                "GenericTriple<String, Int, Bool>",
                'baseValue = "hello"',
                "derivedValue = 100",
                "tripleValue = true",
            ],
        )

        self.expect(
            "frame variable fromNonGeneric",
            substrs=[
                "GenericFromNonGeneric<Int>",
                "baseField = 100",
                "genericField = 42",
            ],
        )

        self.expect(
            "frame variable fromNonGenericDouble",
            substrs=[
                "GenericFromNonGeneric<Double>",
                "baseField = 100",
                "genericField = 3.14",
            ],
        )

        self.expect(
            "frame variable multiDerived",
            substrs=[
                "MultiGenericDerived<Int, Double, Bool>",
                "a = 1",
                "b = 2",
                "c = true",
            ],
        )
