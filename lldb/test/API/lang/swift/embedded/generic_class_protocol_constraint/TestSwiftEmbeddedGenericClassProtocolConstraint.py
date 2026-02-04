import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftEmbeddedGenericClassProtocolConstraint(TestBase):
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")

        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect(
            "frame variable b",
            substrs=[
                "a.SubConcreteB",
                "a.ConcreteB",
                'name = "ConcreteB"',
                "value = 42",
                "extra = 100",
            ],
        )
        self.expect(
            "frame variable a",
            substrs=["a.A<a.SubConcreteB>", "member", "extra = 100", "id = 1"],
        )
        self.expect(
            "frame variable aWithProtocol",
            substrs=["a.A<a.ConcreteB>", "member", "extra = 100", "id = 1"],
        )
        self.expect(
            "frame variable arrayOfB",
            substrs=["[a.B]", "[0] = ", 'name = "ConcreteB"', "value = 42", "[1] = "],
        )
        self.expect(
            "frame variable arrayOfA",
            substrs=[
                "[a.A<a.SubConcreteB>]",
                "[0] = ",
                "[1] = ",
                "member",
                "a.ConcreteB",
                'name = "ConcreteB"',
                "value = 42",
                "extra = 100",
                "id = 1",
            ],
        )

        self.expect(
            "target variable globalB",
            substrs=[
                "a.SubConcreteB",
                "a.ConcreteB",
                'name = "ConcreteB"',
                "value = 42",
                "extra = 100",
            ],
        )
        self.expect(
            "target variable globalA",
            substrs=["a.A<a.SubConcreteB>", "member", "extra = 100", "id = 1"],
        )
        self.expect(
            "target variable globalAWithProtocol",
            substrs=["a.A<a.ConcreteB>", "member", "extra = 100", "id = 1"],
        )
        self.expect(
            "target variable globalArrayOfB",
            substrs=["[a.B]", "[0] = ", 'name = "ConcreteB"', "value = 42", "[1] = "],
        )
        self.expect(
            "target variable globalArrayOfA",
            substrs=[
                "[a.A<a.SubConcreteB>]",
                "[0] = ",
                "[1] = ",
                "member",
                "a.ConcreteB",
                'name = "ConcreteB"',
                "value = 42",
                "extra = 100",
                "id = 1",
            ],
        )
