"""Test the SBDModule APIs."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

import os, signal, subprocess


class SBModuleAPICase(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.background_pid = None

    def tearDown(self):
        TestBase.tearDown(self)
        if self.background_pid:
            os.kill(self.background_pid, signal.SIGKILL)

    @skipIfRemote
    def test_GetObjectName(self):
        """Test the SBModule::GetObjectName() method"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        libfoo_path = self.getBuildArtifact("libfoo.a")
        target_exe = self.dbg.CreateTarget(exe)
        self.assertTrue(target_exe.IsValid(), "Target for a.out is valid")

        # Test that the executable module has no object name (usually the first module in the target)
        exe_module = target_exe.GetModuleAtIndex(0)
        self.assertTrue(exe_module.IsValid(), "Executable module is valid")
        self.assertIsNone(
            exe_module.GetObjectName(), "a.out should have no object name"
        )

        # check archive member names
        module_specs = lldb.SBModuleSpecList.GetModuleSpecifications(libfoo_path)
        self.assertGreater(
            module_specs.GetSize(), 0, "Archive should have at least one module spec"
        )
        found = set()
        expected = {"a.o", "b.o"}
        for i in range(module_specs.GetSize()):
            spec = module_specs.GetSpecAtIndex(i)
            obj_name = spec.GetObjectName()
            self.assertIsInstance(obj_name, str)
            self.assertIn(obj_name, expected, f"Unexpected object name: {obj_name}")
            # create a module from the arhive using the sepc
            module = lldb.SBModule(spec)
            self.assertTrue(module.IsValid(), "Module is valid")
            self.assertTrue(module.IsValid(), f"Module for {obj_name} is valid")
            self.assertEqual(
                module.GetObjectName(), obj_name, f"Object name for {obj_name} matches"
            )
            found.add(obj_name)

        self.assertEqual(found, expected, "Did not find all expected archive members")

    @skipUnlessDarwin
    @skipIfRemote
    def test_module_is_file_backed(self):
        """Test the SBModule::IsFileBacked() method"""
        self.build()
        target, _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        self.assertGreater(target.GetNumModules(), 0)
        main_module = target.GetModuleAtIndex(0)
        self.assertEqual(main_module.GetFileSpec().GetFilename(), "a.out")
        self.assertTrue(
            main_module.IsFileBacked(), "The module should be backed by a file on disk"
        )

        self.dbg.DeleteTarget(target)
        self.assertEqual(self.dbg.GetNumTargets(), 0)

        exe = self.getBuildArtifact("a.out")
        background_process = subprocess.Popen([exe])
        self.assertTrue(background_process, "process is not valid")
        self.background_pid = background_process.pid
        os.unlink(exe)

        target = self.dbg.CreateTarget("")
        self.assertEqual(self.dbg.GetNumTargets(), 1)
        error = lldb.SBError()
        process = target.AttachToProcessWithID(
            self.dbg.GetListener(), self.background_pid, error
        )
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)
        main_module = target.FindModule(lldb.SBFileSpec("a.out"))
        self.assertIsNotNone(main_module)
        self.assertFalse(
            main_module.IsFileBacked(),
            "The module should not be backed by a file on disk.",
        )

        error = process.Destroy()
        self.assertSuccess(
            error, "couldn't destroy process %s" % background_process.pid
        )
