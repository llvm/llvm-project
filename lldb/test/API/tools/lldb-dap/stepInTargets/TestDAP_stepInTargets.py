"""
Test lldb-dap stepInTargets request
"""

from lldbsuite.test.decorators import expectedFailureAll, no_match, skipIf
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import LaunchArgs, StepInTargetsArgs


class TestDAP_stepInTargets(DAPTestCaseBase):
    @expectedFailureAll(oslist=["windows"])
    @skipIf(archs=no_match(["x86_64"]))
    # InstructionControlFlowKind for ARM is not supported yet.
    # On Windows, lldb-dap seems to ignore targetId when stepping into functions.
    # For more context, see https://github.com/llvm/llvm-project/issues/98509.
    def test_basic(self):
        """
        Tests the basic stepping in targets with directly calls.
        """
        program = self.getBuildArtifact("a.out")
        session = self.build_and_create_session()
        source = self.getSourcePath("main.cpp")

        breakpoint_line = line_number(source, "// set breakpoint here")
        # Set breakpoint in the thread function so we can step the threads.
        with session.configure(LaunchArgs(program)) as ctx:
            [bp1] = session.resolve_source_breakpoints(source, [breakpoint_line])
        stop_event = session.verify_stopped_on_breakpoint(bp1, after=ctx.process_event)

        thread_ctxs = session.thread_context_from(stop_event)
        frame_id = thread_ctxs.top_frame().frame.id

        # Request all step in targets list and verify the response.
        step_args = StepInTargetsArgs(frame_id)
        step_in_targets_resp = session.send_request(step_args).result("expect success")
        step_in_targets = step_in_targets_resp.body.targets

        self.assertEqual(len(step_in_targets), 3, "expect 3 step in targets")

        # Verify the target names are correct.
        # The order of funcA and funcB may change depending on the compiler ABI.
        funcA_target = None
        funcB_target = None
        for target in step_in_targets[0:2]:
            if "funcB" in target.label:
                funcB_target = target
            elif "funcA" in target.label:
                funcA_target = target
            else:
                self.fail(f"Unexpected step in target: {target}")

        self.assertIsNotNone(funcA_target, "expect funcA")
        self.assertIsNotNone(funcB_target, "expect funcB")
        self.assertIn("foo", step_in_targets[2].label, "expect foo")

        # Choose to step into second target and verify that we are in the second target,
        # be it funcA or funcB.
        thread_ctxs.step_in(targetId=step_in_targets[1].id)
        top_frame = thread_ctxs.top_frame().frame
        self.assertIsNotNone(top_frame, "expect a top frame")
        self.assertEqual(step_in_targets[1].label, top_frame.name)

        session.continue_to_exit()

    @skipIf(archs=no_match(["x86", "x86_64"]))
    def test_supported_capability_x86_arch(self):
        program = self.getBuildArtifact("a.out")
        source = self.getSourcePath("main.cpp")
        session = self.build_and_create_session()
        bp_lines = [line_number(source, "// set breakpoint here")]
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, bp_lines)

        session.verify_stopped_on_breakpoint(after=ctx.process_event)
        self.assertTrue(
            session.capabilities().supportsStepInTargetsRequest,
            f"expect capability `stepInTarget` is supported with architecture {self.getArchitecture()}",
        )
        session.continue_to_exit()

    @skipIf(archs=["x86", "x86_64"])
    def test_supported_capability_other_archs(self):
        program = self.getBuildArtifact("a.out")
        source = self.getSourcePath("main.cpp")
        session = self.build_and_create_session()
        bp_lines = [line_number(source, "// set breakpoint here")]
        with session.configure(LaunchArgs(program)) as ctx:
            session.resolve_source_breakpoints(source, bp_lines)

        session.verify_stopped_on_breakpoint(after=ctx.process_event)
        self.assertFalse(
            session.capabilities().supportsStepInTargetsRequest,
            f"expect capability `stepInTarget` is not supported with architecture {self.getArchitecture()}",
        )
        session.continue_to_exit()
