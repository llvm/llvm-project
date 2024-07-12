import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StaticInitializers(TestBase):
    @expectedFailureAll(
        archs="aarch64",
        oslist=["freebsd"],
        bugnumber="llvm.org/pr44053",
    )
    def test(self):
        """Test a static initializer."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        # We use counter to observe if the initializer was called.
        self.expect_expr("counter", result_type="int", result_value="0")
        self.expect("expr -p -- struct Foo { Foo() { inc_counter(); } }; Foo f;")
        self.expect_expr("counter", result_type="int", result_value="1")

    def test_failing_init(self):
        """Test a static initializer that fails to execute."""
        self.build()

        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        # FIXME: This error message is not even remotely helpful.
        self.expect(
            "expr -p -- struct Foo2 { Foo2() { do_abort(); } }; Foo2 f;",
            error=True,
            substrs=["error: couldn't run static initializer:"],
        )

    def test_without_process(self):
        """Test a static initializer without a running process."""
        self.expect(
            "expr -p -- int i = 0; struct Foo3 { Foo3() { ++i; } }; Foo3 f;",
            error=True,
            substrs=["Top-level code needs to be inserted into a runnable target"],
        )
