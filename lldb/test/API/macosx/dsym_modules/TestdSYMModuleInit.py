"""
Test that we read in the Python module from a dSYM, and run the
init in debugger and the init in target routines.
"""

import os, shutil

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


@skipUnlessDarwin
class TestdSYMModuleInit(TestBase):
    @no_debug_info_test
    def test_add_module(self):
        """This loads a file into a target and ensures that the python module was
        correctly added and the two initialization functions are called."""
        self.exe_name = "has_dsym"
        self.py_name = self.exe_name + ".py"

        # Now load the target the first time into the debugger:
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.interp = self.dbg.GetCommandInterpreter()

        executable = self.build_dsym(self.exe_name + "_1")
        target = self.createTestTarget(file_path=executable)
        self.check_answers(executable, ["1", "1", "has_dsym_1"])

        # Now make a second target and make sure both get called:
        executable_2 = self.build_dsym(self.exe_name + "_2")
        target_2 = self.createTestTarget(file_path=executable_2)
        self.check_answers(executable_2, ["2", "2", "has_dsym_2"])

    def check_answers(self, name, answers):
        result = lldb.SBCommandReturnObject()
        self.interp.HandleCommand("report_command", result)
        self.assertTrue(
            result.Succeeded(), f"report_command succeeded {result.GetError()}"
        )

        cmd_results = result.GetOutput().split()
        self.assertEqual(answers[0], cmd_results[0], "Right number of module imports")
        self.assertEqual(answers[1], cmd_results[1], "Right number of target notices")
        self.assertIn(answers[2], name, "Right target name")

    def build_dsym(self, name):
        self.build(debug_info="dsym", dictionary={"EXE": name})
        executable = self.getBuildArtifact(name)
        dsym_path = self.getBuildArtifact(name + ".dSYM")
        python_dir_path = dsym_path
        python_dir_path = os.path.join(dsym_path, "Contents", "Resources", "Python")
        if not os.path.exists(python_dir_path):
            os.mkdir(python_dir_path)

        python_file_name = name + ".py"

        module_dest_path = os.path.join(python_dir_path, python_file_name)
        module_origin_path = os.path.join(self.getSourceDir(), self.py_name)
        shutil.copy(module_origin_path, module_dest_path)

        return executable
