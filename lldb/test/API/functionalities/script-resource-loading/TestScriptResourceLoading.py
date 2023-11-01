"""
Test loading python scripting resource from corefile
"""

import os, tempfile

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class ScriptResourceLoadingTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def create_stack_skinny_corefile(self, file):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break", lldb.SBFileSpec("main.cpp")
        )
        self.assertTrue(process.IsValid(), "Process is invalid.")
        # FIXME: Use SBAPI to save the process corefile.
        self.runCmd("process save-core -s stack  " + file)
        self.assertTrue(os.path.exists(file), "No stack-only corefile found.")
        self.assertTrue(self.dbg.DeleteTarget(target), "Couldn't delete target")

    def move_blueprint_to_dsym(self, blueprint_name):
        blueprint_origin_path = os.path.join(self.getSourceDir(), blueprint_name)
        dsym_bundle = self.getBuildArtifact("a.out.dSYM")
        blueprint_destination_path = os.path.join(
            dsym_bundle, "Contents", "Resources", "Python"
        )
        if not os.path.exists(blueprint_destination_path):
            os.mkdir(blueprint_destination_path)

        blueprint_destination_path = os.path.join(
            blueprint_destination_path, "a_out.py"
        )
        shutil.copy(blueprint_origin_path, blueprint_destination_path)

    @skipUnlessDarwin
    def test_script_resource_loading(self):
        """
        Test that we're able to load the python scripting resource from
        corefile dSYM bundle.

        """
        self.build()

        self.runCmd("settings set target.load-script-from-symbol-file true")
        self.move_blueprint_to_dsym("my_scripting_resource.py")

        corefile_process = None
        with tempfile.NamedTemporaryFile() as file:
            self.create_stack_skinny_corefile(file.name)
            corefile_target = self.dbg.CreateTarget(None)
            corefile_process = corefile_target.LoadCore(
                self.getBuildArtifact(file.name)
            )
        self.assertTrue(corefile_process, PROCESS_IS_VALID)
        self.expect("command script list", substrs=["test_script_resource_loading"])
        self.runCmd("test_script_resource_loading")
