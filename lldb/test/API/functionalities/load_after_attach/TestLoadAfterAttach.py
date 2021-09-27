import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote
    def test_load_after_attach(self):
        self.build()

        ctx = self.platformContext
        lib_name = ctx.shlib_prefix + 'lib_b.' + ctx.shlib_extension

        exe = self.getBuildArtifact("a.out")
        lib = self.getBuildArtifact(lib_name)

        # Spawn a new process.
        # use realpath to workaround llvm.org/pr48376
        # Pass path to solib for dlopen to properly locate the library.
        popen = self.spawnSubprocess(os.path.realpath(exe), args = [os.path.realpath(lib)])
        pid = popen.pid

        # Attach to the spawned process.
        self.runCmd("process attach -p " + str(pid))

        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        self.assertTrue(process, PROCESS_IS_VALID)

        # Continue until first breakpoint.
        breakpoint1 = self.target().BreakpointCreateBySourceRegex(
            "// break here", lldb.SBFileSpec("main.cpp"))
        self.assertEqual(breakpoint1.GetNumResolvedLocations(), 1)
        stopped_threads = lldbutil.continue_to_breakpoint(self.process(), breakpoint1)
        self.assertEqual(len(stopped_threads), 1)

        # Check that image list does not contain liblib_b before dlopen.
        self.match(
                "image list",
                patterns = [lib_name],
                matching = False,
                msg = lib_name + " should not have been in image list")

        # Change a variable to escape the loop
        self.runCmd("expression main_thread_continue = 1")

        # Continue so that dlopen is called.
        breakpoint2 = self.target().BreakpointCreateBySourceRegex(
            "// break after dlopen", lldb.SBFileSpec("main.cpp"))
        self.assertEqual(breakpoint2.GetNumResolvedLocations(), 1)
        stopped_threads = lldbutil.continue_to_breakpoint(self.process(), breakpoint2)
        self.assertEqual(len(stopped_threads), 1)

        # Check that image list contains liblib_b after dlopen.
        self.match(
                "image list",
                patterns = [lib_name],
                matching = True,
                msg = lib_name + " missing in image list")

