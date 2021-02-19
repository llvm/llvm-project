import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def build_launch_and_attach(self):
        self.build()
        # launch
        exe = self.getBuildArtifact("a.out")
        popen = self.spawnSubprocess(exe)
        # attach
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        listener = lldb.SBListener("my.attach.listener")
        error = lldb.SBError()
        process = target.AttachToProcessWithID(listener, popen.pid, error)
        self.assertTrue(error.Success() and process, PROCESS_IS_VALID)
        return process

    def assertModuleIsLoaded(self, module_name):
        feature_module = self.dbg.GetSelectedTarget().FindModule(lldb.SBFileSpec(module_name))
        self.assertTrue(feature_module.IsValid(), f"Module {module_name} should be loaded")

    def assertModuleIsNotLoaded(self, module_name):
        feature_module = self.dbg.GetSelectedTarget().FindModule(lldb.SBFileSpec(module_name))
        self.assertFalse(feature_module.IsValid(), f"Module {module_name} should not be loaded")

    @skipIfRemote
    @skipUnlessLinux
    @no_debug_info_test
    def test(self):
        '''
            This test makes sure that after attach lldb still gets notifications
            about new modules being loaded by the process
        '''
        process = self.build_launch_and_attach()
        thread = process.GetSelectedThread()
        self.assertModuleIsNotLoaded("libfeature.so")
        thread.GetSelectedFrame().EvaluateExpression("flip_to_1_to_continue = 1")
        # Continue so that dlopen is called.
        breakpoint = self.target().BreakpointCreateBySourceRegex(
            "// break after dlopen", lldb.SBFileSpec("main.c"))
        self.assertNotEqual(breakpoint.GetNumResolvedLocations(), 0)
        stopped_threads = lldbutil.continue_to_breakpoint(self.process(), breakpoint)
        self.assertModuleIsLoaded("libfeature.so")
