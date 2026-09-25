"""
Test lldb-dap saving a core minidump file and attaching to it.
"""

from pathlib import Path

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import line_number
from lldbsuite.test.tools.lldb_dap import DAPTestCaseBase
from lldbsuite.test.tools.lldb_dap.types import AttachArgs, LaunchArgs


class TestDAP_save_core(DAPTestCaseBase):
    SHARED_BUILD_TESTCASE = False

    @skipUnlessArch("x86_64")
    @requireLinux
    def test_save_and_reload_core(self):
        """Save minidump cores in every supported `--style` from a single
        live stop, then re-attach to each and verify the frame, thread
        count, and module count all round-trip."""

        session = self.build_and_create_session()
        source = "main.cpp"
        program = self.getBuildArtifact("a.out")
        breakpoint_line = line_number(source, "// breakpoint 1")

        with session.configure(LaunchArgs(program)) as cm:
            [bp_id] = session.resolve_source_breakpoints(source, [breakpoint_line])

        stop_event = session.verify_stopped_on_breakpoint(bp_id, after=cm.process_event)

        # Snapshot the live state so we can assert the reloaded core matches.
        thread_count = len(session.get_threads())
        module_count = len(session.get_modules())

        core_styles = ["stack", "modified-memory", "full"]
        top_frame = session.top_frame_from(stop_event)
        for style in core_styles:
            path = Path(self.getBuildArtifact(f"core.{style}.dmp"))
            self.assertFalse(path.exists(), f"stale core file: {path}")

            save_core = "process save-core --plugin-name=minidump"
            top_frame.evaluate(f"`{save_core} --style={style} {path}", context="repl")
            self.assertTrue(path.is_file(), f"{style} core file is a file")

            with self.subTest(style=style):
                self.verify_core(style, path, module_count, thread_count)

        session.continue_to_exit(exitCode=3)

    def verify_core(
        self, style: str, core_path: Path, module_count: int, thread_count: int
    ):
        """Attach to a saved core and verify the reloaded process state
        matches what was captured live: current frame, thread count,
        module count."""
        session = self.create_session(adapter=self.create_stdio_debug_adapter())
        process_event = session.attach(AttachArgs(coreFile=str(core_path)))

        stop_event = session.verify_stopped_on_exception(after=process_event)
        top_frame = session.top_frame_from(stop_event).frame
        self.assertTrue(
            top_frame.name.startswith("function"),
            "expected to stop inside `function`",
        )

        expected_line = line_number("main.cpp", "// breakpoint 1")
        self.assertEqual(top_frame.line, expected_line)

        core_thread_count = len(session.get_threads())
        core_module_count = len(session.get_modules())

        self.assertEqual(core_thread_count, thread_count)
        if style != "full":  # FIXME: There is a bug on linux
            self.assertEqual(core_module_count, module_count)
