"""
Test saving core minidump from lldb-dap
"""

import dap_server
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbdap_testcase
from lldbsuite.test import lldbutil


class TestDAP_save_core(lldbdap_testcase.DAPTestCaseBase):
    @skipUnlessArch("x86_64")
    @skipUnlessPlatform(["linux"])
    def test_save_core(self):
        """
        Tests saving core minidump from lldb-dap.
        """
        program = self.getBuildArtifact("a.out")
        self.build_and_launch(program)
        source = "main.cpp"
        # source_path = os.path.join(os.getcwd(), source)
        breakpoint1_line = line_number(source, "// breakpoint 1")
        lines = [breakpoint1_line]
        # Set breakpoint in the thread function so we can step the threads
        breakpoint_ids = self.set_source_breakpoints(source, lines)
        self.assertEqual(
            len(breakpoint_ids), len(lines), "expect correct number of breakpoints"
        )
        self.continue_to_breakpoints(breakpoint_ids)

        # Getting dap stack trace may trigger __lldb_caller_function JIT module to be created.
        self.get_stackFrames(startFrame=0)

        # Evaluating an expression that cause "_$__lldb_valid_pointer_check" JIT module to be created.
        expression = 'printf("this is a test")'
        self.dap_server.request_evaluate(expression, context="watch")

        # Verify "_$__lldb_valid_pointer_check" JIT module is created.
        modules = self.dap_server.get_modules()
        self.assertTrue(modules["_$__lldb_valid_pointer_check"])
        thread_count = len(self.dap_server.get_threads())

        core_stack = self.getBuildArtifact("core.stack.dmp")
        core_dirty = self.getBuildArtifact("core.dirty.dmp")
        core_full = self.getBuildArtifact("core.full.dmp")

        base_command = "`process save-core --plugin-name=minidump "
        self.dap_server.request_evaluate(
            base_command + " --style=stack '%s'" % (core_stack), context="repl"
        )

        self.assertTrue(os.path.isfile(core_stack))
        self.verify_core_file(core_stack, len(modules), thread_count)

        self.dap_server.request_evaluate(
            base_command + " --style=modified-memory '%s'" % (core_dirty),
            context="repl",
        )
        self.assertTrue(os.path.isfile(core_dirty))
        self.verify_core_file(core_dirty, len(modules), thread_count)

        self.dap_server.request_evaluate(
            base_command + " --style=full '%s'" % (core_full), context="repl"
        )
        self.assertTrue(os.path.isfile(core_full))
        self.verify_core_file(core_full, len(modules), thread_count)

    def verify_core_file(self, core_path, expected_module_count, expected_thread_count):
        # To verify, we'll launch with the mini dump
        target = self.dbg.CreateTarget(None)
        process = target.LoadCore(core_path)

        # check if the core is in desired state
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertTrue(process.GetProcessInfo().IsValid())
        self.assertNotEqual(target.GetTriple().find("linux"), -1)
        self.assertTrue(target.GetNumModules(), expected_module_count)
        self.assertEqual(process.GetNumThreads(), expected_thread_count)
