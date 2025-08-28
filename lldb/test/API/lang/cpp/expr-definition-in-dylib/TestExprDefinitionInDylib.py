import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprDefinitionInDylibTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIfWindows
    def test(self):
        """
        Tests that we can call functions whose definition
        is in a different LLDB module than it's declaration.
        """
        self.build()

        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        env = self.registerSharedLibrariesWithTarget(target, ["lib"])

        breakpoint = lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", line_number("main.cpp", "return")
        )

        process = target.LaunchSimple(None, env, self.get_process_working_directory())

        self.assertIsNotNone(
            lldbutil.get_one_thread_stopped_at_breakpoint_id(self.process(), breakpoint)
        )

        self.expect_expr("f.method()", result_value="-72", result_type="int")
        self.expect_expr("Foo()", result_type="Foo")

        # FIXME: mangled name lookup for ABI-tagged ctors fails because
        # the debug-info AST doesn't have ABI-tag information.
        self.expect(
            "expr Bar()", error=True, substrs=["error: Couldn't look up symbols"]
        )
