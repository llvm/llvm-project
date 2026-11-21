"""
Test that we read in the Python module from a dSYM, and run the
init in debugger and the init in target routines.
"""

import json
import os, shutil

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *


@requireDarwin
class TestdSYMModuleInit(TestBase):
    SHARED_BUILD_TESTCASE = False

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

    @no_debug_info_test
    def test_disabled_autoload_keeps_symbols_lazy(self):
        self.check_autoload_policy("false")

    @no_debug_info_test
    def test_disabled_module_override_keeps_symbols_lazy(self):
        self.check_autoload_policy("true", "false")

    @no_debug_info_test
    def test_enabled_module_override_loads_script(self):
        self.check_autoload_policy("false", "true")

    def check_autoload_policy(self, default_policy, module_policy=None):
        self.py_name = "has_dsym.py"
        name = f"has_dsym_1_{default_policy}_{module_policy}"
        executable = self.build_dsym(name)

        self.runCmd("settings set target.preload-symbols false")
        self.runCmd(
            f"settings set target.load-script-from-symbol-file {default_policy}"
        )
        if module_policy is not None:
            self.runCmd(
                "settings set target.auto-load-scripts-for-modules "
                f"{name}={module_policy}"
            )
        self.interp = self.dbg.GetCommandInterpreter()
        target = self.createTestTarget(
            file_path=executable, load_dependent_modules=False
        )

        if module_policy == "true":
            self.check_answers(executable, ["1", "1", name])
            return

        # GetStatistics does not create the symbol file. Inspecting symbols
        # through SBModule would itself defeat the laziness being tested.
        stream = lldb.SBStream()
        target.GetStatistics().GetAsJSON(stream)
        stats = json.loads(stream.GetData())
        module = next(m for m in stats["modules"] if m["path"] == executable)
        self.assertNotIn("symbolFilePath", module)
        self.assertEqual(module["debugInfoByteSize"], 0)

        # Disabling script autoload must not disable subsequent source lookup.
        breakpoint = target.BreakpointCreateByLocation(
            "main.c", line_number("main.c", "Set a breakpoint here")
        )
        self.assertEqual(breakpoint.GetNumLocations(), 1)
        stream = lldb.SBStream()
        target.GetStatistics().GetAsJSON(stream)
        stats = json.loads(stream.GetData())
        module = next(m for m in stats["modules"] if m["path"] == executable)
        self.assertIn("symbolFilePath", module)
        self.assertGreater(module["debugInfoByteSize"], 0)

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
