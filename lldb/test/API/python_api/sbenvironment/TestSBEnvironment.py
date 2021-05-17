"""Test the SBEnvironment APIs."""



from math import fabs
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class SBEnvironmentAPICase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    # We use this function to test both kind of accessors:
    # .  Get*AtIndex and GetEntries
    def assertEqualEntries(self, env, entries):
        self.assertEqual(env.GetNumValues(), len(entries))
        for i in range(env.GetNumValues()):
            name = env.GetNameAtIndex(i)
            value = env.GetValueAtIndex(i)
            self.assertIn(name + "=" + value, entries)

        entries = env.GetEntries()
        self.assertEqual(entries.GetSize(), len(entries))
        for i in range(entries.GetSize()):
            (name, value) = entries.GetStringAtIndex(i).split("=")
            self.assertIn(name + "=" + value, entries)



    @skipIfRemote # Remote environment not supported.
    def test_platform_environment(self):
        env = self.dbg.GetSelectedPlatform().GetEnvironment()
        # We assume at least PATH is set
        self.assertNotEqual(env.Get("PATH"), None)


    def test_launch_info(self):
        target = self.dbg.CreateTarget("")
        launch_info = target.GetLaunchInfo()
        env = launch_info.GetEnvironment()
        env_count = env.GetNumValues()

        env.Set("FOO", "bar", overwrite=True)
        self.assertEqual(env.GetNumValues(), env_count + 1)

        # Make sure we only modify the copy of the launchInfo's environment
        self.assertEqual(launch_info.GetEnvironment().GetNumValues(), env_count)

        launch_info.SetEnvironment(env, append=True)
        self.assertEqual(launch_info.GetEnvironment().GetNumValues(), env_count + 1)

        env.Set("FOO", "baz", overwrite=True)
        launch_info.SetEnvironment(env, append=True)
        self.assertEqual(launch_info.GetEnvironment().GetNumValues(), env_count + 1)
        self.assertEqual(launch_info.GetEnvironment().Get("FOO"), "baz")

        # Make sure we can replace the launchInfo's environment
        env.Clear()
        env.Set("BAR", "foo", overwrite=True)
        env.PutEntry("X=y")
        launch_info.SetEnvironment(env, append=False)
        self.assertEqualEntries(launch_info.GetEnvironment(), ["BAR=foo", "X=y"])


    @skipIfRemote # Remote environment not supported.
    def test_target_environment(self):
        env = self.dbg.GetSelectedTarget().GetEnvironment()
        # There is no target, so env should be empty
        self.assertEqual(env.GetNumValues(), 0)
        self.assertEqual(env.Get("PATH"), None)

        target = self.dbg.CreateTarget("")
        env = target.GetEnvironment()
        path = env.Get("PATH")
        # Now there's a target, so at least PATH should exist
        self.assertNotEqual(path, None)

        # Make sure we are getting a copy by modifying the env we just got
        env.PutEntry("PATH=#" + path)
        self.assertEqual(target.GetEnvironment().Get("PATH"), path)

    def test_creating_and_modifying_environment(self):
        env = lldb.SBEnvironment()

        self.assertEqual(env.Get("FOO"), None)
        self.assertEqual(env.Get("BAR"), None)

        # We also test empty values
        self.assertTrue(env.Set("FOO", "", overwrite=False))
        env.Set("BAR", "foo", overwrite=False)

        self.assertEqual(env.Get("FOO"), "")
        self.assertEqual(env.Get("BAR"), "foo")

        self.assertEqual(env.GetNumValues(), 2)

        self.assertEqualEntries(env, ["FOO=", "BAR=foo"])

        # Make sure modifications work
        self.assertFalse(env.Set("FOO", "bar", overwrite=False))
        self.assertEqual(env.Get("FOO"), "")

        env.PutEntry("FOO=bar")
        self.assertEqual(env.Get("FOO"), "bar")

        self.assertEqualEntries(env, ["FOO=bar", "BAR=foo"])

        # Make sure we can unset
        self.assertTrue(env.Unset("FOO"))
        self.assertFalse(env.Unset("FOO"))
        self.assertEqual(env.Get("FOO"), None)

        # Test SetEntries
        entries = lldb.SBStringList()
        entries.AppendList(["X=x", "Y=y"], 2)

        env.SetEntries(entries, append=True)
        self.assertEqualEntries(env, ["BAR=foo", "X=x", "Y=y"])

        env.SetEntries(entries, append=False)
        self.assertEqualEntries(env, ["X=x", "Y=y"])

        entries.Clear()
        entries.AppendList(["X=y", "Y=x"], 2)
        env.SetEntries(entries, append=True)
        self.assertEqualEntries(env, ["X=y", "Y=x"])

        # Test clear
        env.Clear()
        self.assertEqualEntries(env, [])
