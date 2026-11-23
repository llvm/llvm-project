"""
Test that Process::LoadImage reports a real error, not an empty/null one, on
every call after the first, when there is no live process to call dlopen in
(e.g. debugging a core file).
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ProcessLoadErrorPostmortemTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(archs=no_match(["x86_64", "arm64", "arm64e", "aarch64"]))
    @skipIfWindows  # Windows does not support "process load" the same way.
    @skipIfRemote  # save-core / core loading is local-target only here.
    def test_load_image_error_replayed_on_core(self):
        self.build()
        exe = self.getBuildArtifact("a.out")
        core = self.getBuildArtifact("process.core")

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.runCmd("process save-core --style=stack " + core)
        process.Kill()
        self.dbg.DeleteTarget(target)

        target = self.dbg.CreateTarget(exe)
        process = target.LoadCore(core)
        self.assertTrue(process.IsValid(), "Could not load the core file")

        nonexistent = lldb.SBFileSpec(
            "/NoSuchDir/NoSuchSubdir/not-a-real-library", False
        )

        messages = []
        for _ in range(2):
            error = lldb.SBError()
            token = process.LoadImage(nonexistent, error)
            self.assertEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN)
            self.assertTrue(error.Fail())
            message = error.GetCString()
            self.assertIsNotNone(message)
            self.assertNotEqual(message, "")
            self.assertNotIn("(null)", message)
            messages.append(message)

        self.assertEqual(messages[0], messages[1])
