"""
Test when LLDB runs the hooks of a Python module in a dSYM.
"""

import os, shutil

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


@requireDarwin
class TestdSYMModuleHooks(TestBase):
    SHARED_BUILD_TESTCASE = False
    COUNTS = (
        "module_hooks_inits",
        "module_hooks_targets",
        "module_hooks_debugger_ids",
        "module_hooks_added_targets",
        "failing_hooks_inits",
        "read_header_runs",
        "read_header_magic",
    )
    MH_MAGIC_64 = 0xFEEDFACF

    def setUp(self):
        TestBase.setUp(self)
        for name in self.COUNTS:
            if hasattr(lldb, name):
                delattr(lldb, name)

    def counts(self):
        return (
            getattr(lldb, "module_hooks_inits", 0),
            getattr(lldb, "module_hooks_targets", 0),
        )

    @no_debug_info_test
    def test_scripts_run_once_per_target(self):
        """The module is seen again when the process loads it and gives it a
        load address, but its script only runs once for the target."""
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.build_dsym("has_hooks", "module_hooks.py")
        log = self.enable_script_log()
        lldbutil.run_to_source_breakpoint(
            self,
            "Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            exe_name="has_hooks",
        )
        self.assertEqual(self.counts(), (1, 1))
        self.assertRegex(
            self.read_script_log(log),
            r"Skipping the scripts of '.*': this target already loaded them",
        )

    @no_debug_info_test
    def test_second_target_runs_scripts_once(self):
        """A second target that loads the module runs its script once too, and
        __lldb_module_added_to_target gets that target."""
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.build_dsym("has_hooks", "module_hooks.py")
        targets = []
        for _ in range(2):
            target, process, _, _ = lldbutil.run_to_source_breakpoint(
                self,
                "Set a breakpoint here",
                lldb.SBFileSpec("main.c"),
                exe_name="has_hooks",
            )
            targets.append(target)
            process.Kill()
        self.assertEqual(self.counts(), (2, 2))
        self.assertEqual(lldb.module_hooks_added_targets, targets)

    @no_debug_info_test
    def test_second_debugger_runs_init_module_once(self):
        """A second debugger that loads the module runs __lldb_init_module once,
        with that debugger."""
        self.runCmd("settings set target.load-script-from-symbol-file true")
        exe = self.build_dsym("has_hooks", "module_hooks.py")
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self,
            "Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            exe_name="has_hooks",
        )
        process.Kill()

        debugger = lldb.SBDebugger.Create(False)
        self.addTearDownHook(lambda: lldb.SBDebugger.Destroy(debugger))
        debugger.SetAsync(False)
        debugger.HandleCommand("settings set target.load-script-from-symbol-file true")
        target = debugger.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        target.BreakpointCreateBySourceRegex(
            "Set a breakpoint here", lldb.SBFileSpec("main.c")
        )
        process = target.LaunchSimple(None, None, self.get_process_working_directory())
        self.assertState(process.GetState(), lldb.eStateStopped)
        process.Kill()

        self.assertEqual(
            lldb.module_hooks_debugger_ids, [self.dbg.GetID(), debugger.GetID()]
        )

    @no_debug_info_test
    def test_reload_module_scripts_setting(self):
        """target.experimental.reload-module-scripts brings back loading the
        script every time the target sees the module."""
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.runCmd("settings set target.experimental.reload-module-scripts true")
        self.addTearDownHook(
            lambda: self.runCmd(
                "settings set target.experimental.reload-module-scripts false"
            )
        )
        self.build_dsym("has_hooks", "module_hooks.py")
        lldbutil.run_to_source_breakpoint(
            self,
            "Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            exe_name="has_hooks",
        )
        inits, targets = self.counts()
        self.assertGreater(inits, 1)
        self.assertGreater(targets, 1)

    @no_debug_info_test
    def test_failing_script_runs_once_per_target(self):
        """We record the module before loading its script, so a script that
        fails to load isn't tried again when the module is seen again."""
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.build_dsym("has_failing_hooks", "failing_hooks.py")
        lldbutil.run_to_source_breakpoint(
            self,
            "Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            exe_name="has_failing_hooks",
        )
        self.assertEqual(getattr(lldb, "failing_hooks_inits", 0), 1)

    @no_debug_info_test
    def test_one_shot_stop_hook_reads_module(self):
        """A one-shot stop hook that __lldb_module_added_to_target adds reads
        the module from memory at the first stop where it is loaded, and LLDB
        deletes it afterwards."""
        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.build_dsym("reads_header", "read_header_hook.py")
        lldbutil.run_to_source_breakpoint(
            self,
            "Set a breakpoint here",
            lldb.SBFileSpec("main.c"),
            exe_name="reads_header",
        )
        self.assertEqual(getattr(lldb, "read_header_magic", None), self.MH_MAGIC_64)
        self.expect("target stop-hook list", substrs=["No stop hooks."])

    def enable_script_log(self):
        log = self.getBuildArtifact("script.log")
        self.runCmd(f"log enable -f {log} lldb script")
        return log

    def read_script_log(self, log):
        self.runCmd("log disable lldb script")
        with open(log) as f:
            return f.read()

    def build_dsym(self, name, script):
        self.build(debug_info="dsym", dictionary={"EXE": name})
        executable = self.getBuildArtifact(name)
        python_dir_path = os.path.join(
            self.getBuildArtifact(name + ".dSYM"), "Contents", "Resources", "Python"
        )
        if not os.path.exists(python_dir_path):
            os.mkdir(python_dir_path)
        shutil.copy(
            os.path.join(self.getSourceDir(), script),
            os.path.join(python_dir_path, name + ".py"),
        )
        return executable
