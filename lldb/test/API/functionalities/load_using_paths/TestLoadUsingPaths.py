"""
Test that SBProcess.LoadImageUsingPaths works correctly.
"""



import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipIfWindows  # The Windows platform doesn't implement DoLoadImage.
class LoadUsingPathsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Make the hidden directory in the build hierarchy:
        lldbutil.mkdir_p(self.getBuildArtifact("hidden"))

        # Invoke the default build rule.
        self.build()

        ext = 'so'
        if self.platformIsDarwin():
            ext = 'dylib'
        self.lib_name = 'libloadunload.' + ext

        self.wd = os.path.realpath(self.getBuildDir())
        self.hidden_dir = os.path.join(self.wd, 'hidden')
        self.hidden_lib = os.path.join(self.hidden_dir, self.lib_name)

    @skipIfRemote
    @skipIfWindows  # Windows doesn't have dlopen and friends, dynamic libraries work differently
    @expectedFlakeyNetBSD
    @expectedFailureAll(oslist=["linux"], archs=['arm'], bugnumber="llvm.org/pr45894")
    def test_load_using_paths(self):
        """Test that we can load a module by providing a set of search paths."""
        if self.platformIsDarwin():
            dylibName = 'libloadunload_d.dylib'
        else:
            dylibName = 'libloadunload_d.so'

        # The directory with the dynamic library we did not link to.
        path_dir = os.path.join(self.getBuildDir(), "hidden")

        (target, process, thread,
         _) = lldbutil.run_to_source_breakpoint(self,
                                                "Break here to do the load using paths",
                                                lldb.SBFileSpec("main.cpp"))
        error = lldb.SBError()
        lib_spec = lldb.SBFileSpec(self.lib_name)
        paths = lldb.SBStringList()
        paths.AppendString(self.wd)
        paths.AppendString(os.path.join(self.wd, "no_such_dir"))

        out_spec = lldb.SBFileSpec()

        # First try with no correct directories on the path, and make sure that doesn't blow up:
        token = process.LoadImageUsingPaths(lib_spec, paths, out_spec, error)
        self.assertEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Only looked on the provided path.")
        # Make sure we got some error back in this case.  Since we don't actually know what
        # the error will look like, let's look for the absence of "unknown reasons".
        error_str = error.description
        self.assertNotEqual(len(error_str), 0, "Got an empty error string")
        self.assertNotIn("unknown reasons", error_str, "Error string had unknown reasons")
        
        # Now add the correct dir to the paths list and try again:
        paths.AppendString(self.hidden_dir)
        token = process.LoadImageUsingPaths(lib_spec, paths, out_spec, error)

        self.assertNotEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Got a valid token")
        self.assertEqual(out_spec, lldb.SBFileSpec(self.hidden_lib), "Found the expected library")

        # Make sure this really is in the image list:
        loaded_module = target.FindModule(out_spec)

        self.assertTrue(loaded_module.IsValid(), "The loaded module is in the image list.")

        # Now see that we can call a function in the loaded module.
        value = thread.frames[0].EvaluateExpression("d_function()", lldb.SBExpressionOptions())
        self.assertSuccess(value.GetError(), "Got a value from the expression")
        ret_val = value.GetValueAsSigned()
        self.assertEqual(ret_val, 12345, "Got the right value")

        # Make sure the token works to unload it:
        process.UnloadImage(token)

        # Make sure this really is no longer in the image list:
        loaded_module = target.FindModule(out_spec)

        self.assertFalse(loaded_module.IsValid(), "The unloaded module is no longer in the image list.")

        # Make sure a relative path also works:
        paths.Clear()
        paths.AppendString(os.path.join(self.wd, "no_such_dir"))
        paths.AppendString(self.wd)
        relative_spec = lldb.SBFileSpec(os.path.join("hidden", self.lib_name))

        out_spec = lldb.SBFileSpec()
        token = process.LoadImageUsingPaths(relative_spec, paths, out_spec, error)

        self.assertNotEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Got a valid token with relative path")
        self.assertEqual(out_spec, lldb.SBFileSpec(self.hidden_lib), "Found the expected library with relative path")

        process.UnloadImage(token)

        # Make sure the presence of an empty path doesn't mess anything up:
        paths.Clear()
        paths.AppendString("")
        paths.AppendString(os.path.join(self.wd, "no_such_dir"))
        paths.AppendString(self.wd)
        relative_spec = lldb.SBFileSpec(os.path.join("hidden", self.lib_name))

        out_spec = lldb.SBFileSpec()
        token = process.LoadImageUsingPaths(relative_spec, paths, out_spec, error)

        self.assertNotEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Got a valid token with included empty path")
        self.assertEqual(out_spec, lldb.SBFileSpec(self.hidden_lib), "Found the expected library with included empty path")

        process.UnloadImage(token)

        # Finally, passing in an absolute path should work like the basename:
        # This should NOT work because we've taken hidden_dir off the paths:
        abs_spec = lldb.SBFileSpec(os.path.join(self.hidden_dir, self.lib_name))

        token = process.LoadImageUsingPaths(lib_spec, paths, out_spec, error)
        self.assertEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Only looked on the provided path.")

        # But it should work when we add the dir:
        # Now add the correct dir to the paths list and try again:
        paths.AppendString(self.hidden_dir)
        token = process.LoadImageUsingPaths(lib_spec, paths, out_spec, error)

        self.assertNotEqual(token, lldb.LLDB_INVALID_IMAGE_TOKEN, "Got a valid token")
        self.assertEqual(out_spec, lldb.SBFileSpec(self.hidden_lib), "Found the expected library")
