import lldb
import subprocess
import os
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestMSVCRTCException(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessPlatform(["windows"])
    @skipUnlessMSVC
    def test_msvc_runtime_checks(self):
        """Test that lldb prints MSVC's runtime checks exceptions as stop reasons."""

        src = os.path.join(self.getSourceDir(), "main.c")
        exe = os.path.join(self.getBuildDir(), "a.exe")

        result = subprocess.run(
            ["cl.exe", "/nologo", "/Od", "/Zi", "/MDd", "/RTC1", "/Fe" + exe, src],
            cwd=self.getBuildDir(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            "Compilation failed:\n" + result.stdout + result.stderr,
        )

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target.IsValid(), "Could not create target")

        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertTrue(process.IsValid(), "Could not launch process")

        self.assertEqual(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonException)
        self.assertIsNotNone(thread, "No thread stopped with exception stop reason")

        stop_description = thread.GetStopDescription(256)
        self.assertIn(
            "Run-time check failure",
            stop_description,
            "Stop reason does not mention run-time check failure",
        )
        self.assertIn(
            "variable 'x' is being used without being initialized",
            stop_description,
            "Stop reason does not mention uninitialized variable 'x'",
        )
