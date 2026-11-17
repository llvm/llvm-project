"""
Test the dSYM scripts of the images that a scripted process reports.
"""

import json, os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


@skipUnlessDarwin
class TestScriptedProcessdSYMScripts(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    RECORDS = ("payload_inits", "payload_targets", "payload_magics")
    LOAD_ADDR = 0x100000000
    MH_MAGIC_64 = 0xFEEDFACF

    def setUp(self):
        TestBase.setUp(self)
        for name in self.RECORDS:
            if hasattr(lldb, name):
                delattr(lldb, name)

    def test_scripts_run_once_per_target(self):
        """A scripted process adds the image it reports to its target, then
        gives it a load address, so the target sees the image twice. Its dSYM
        script still only runs once for each target that loads it."""
        self.launch_scripted_processes(2)
        self.assertEqual(lldb.payload_inits, 2)
        self.assertEqual(lldb.payload_targets, [0, 1])

    def test_one_shot_stop_hook_reads_image(self):
        """The image's dSYM script adds a one-shot stop hook to each target,
        which reads the image's Mach-O header through the scripted process at
        its first stop. LLDB deletes the hook afterwards."""
        targets = self.launch_scripted_processes(2)
        self.assertEqual(
            lldb.payload_magics, [(0, self.MH_MAGIC_64), (1, self.MH_MAGIC_64)]
        )
        for target in targets:
            result = lldb.SBCommandReturnObject()
            self.dbg.GetCommandInterpreter().HandleCommand(
                "target stop-hook list", lldb.SBExecutionContext(target), result
            )
            self.assertCommandReturn(result)
            self.assertIn("No stop hooks.", result.GetOutput())

    def launch_scripted_processes(self, count):
        """Launch count scripted processes, each in its own target, that all
        report libpayload.dylib."""
        self.build()
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), "image_scripted_process.py")
        )
        return [self.launch_scripted_process() for _ in range(count)]

    def launch_scripted_process(self):
        error = lldb.SBError()
        target = self.dbg.CreateTarget(
            None, f"{self.getArchitecture()}-apple-macosx", None, False, error
        )
        self.assertSuccess(error)

        args = lldb.SBStructuredData()
        args.SetFromJSON(
            json.dumps(
                {
                    "path": self.getBuildArtifact("libpayload.dylib"),
                    "load_addr": self.LOAD_ADDR,
                }
            )
        )
        launch_info = lldb.SBLaunchInfo(None)
        launch_info.SetProcessPluginName("ScriptedProcess")
        launch_info.SetScriptedProcessClassName(
            "image_scripted_process.ImageScriptedProcess"
        )
        launch_info.SetScriptedProcessDictionary(args)
        process = target.Launch(launch_info, error)
        self.assertSuccess(error)
        self.assertState(process.GetState(), lldb.eStateStopped)
        return target
