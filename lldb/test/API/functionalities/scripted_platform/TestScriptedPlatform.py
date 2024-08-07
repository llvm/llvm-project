"""
Test python scripted platform in lldb
"""

import os, shutil

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class ScriptedPlatformTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_python_plugin_package(self):
        """Test that the lldb python module has a `plugins.scripted_platform`
        package."""
        self.expect(
            "script import lldb.plugins",
            substrs=["ModuleNotFoundError"],
            matching=False,
        )

        self.expect("script dir(lldb.plugins)", substrs=["scripted_platform"])

        self.expect(
            "script import lldb.plugins.scripted_platform",
            substrs=["ModuleNotFoundError"],
            matching=False,
        )

        self.expect(
            "script dir(lldb.plugins.scripted_platform)", substrs=["ScriptedPlatform"]
        )

        self.expect(
            "script from lldb.plugins.scripted_platform import ScriptedPlatform",
            substrs=["ImportError"],
            matching=False,
        )

        self.expect(
            "script dir(ScriptedPlatform)",
            substrs=[
                "attach_to_process",
                "kill_process",
                "launch_process",
                "list_processes",
            ],
        )

    @skipUnlessDarwin
    def test_list_processes(self):
        """Test that we can load and select an lldb scripted platform using the
        SBAPI, check its process ID, parent, name & triple.
        """
        os.environ["SKIP_SCRIPTED_PLATFORM_SELECT"] = "1"

        def cleanup():
            del os.environ["SKIP_SCRIPTED_PLATFORM_SELECT"]

        self.addTearDownHook(cleanup)

        scripted_platform_example_relpath = "my_scripted_platform.py"
        self.runCmd(
            "command script import "
            + os.path.join(self.getSourceDir(), scripted_platform_example_relpath)
        )

        proc_info = {}
        proc_info["name"] = "a.out"
        proc_info["arch"] = "arm64-apple-macosx"
        proc_info["pid"] = 420
        proc_info["parent"] = 42
        proc_info["uid"] = 501
        proc_info["gid"] = 20

        structured_data = lldb.SBStructuredData()
        structured_data.SetFromJSON(json.dumps({"processes": [proc_info]}))

        err = lldb.SBError()
        platform = lldb.SBPlatform(
            "scripted-platform",
            self.dbg,
            "my_scripted_platform.MyScriptedPlatform",
            structured_data,
            err,
        )

        self.assertSuccess(err)
        self.assertTrue(platform and platform.IsValid())
        self.assertTrue(platform.IsConnected)

        err = lldb.SBError()
        proc_list = platform.GetAllProcesses(err)
        self.assertSuccess(err)
        self.assertEqual(proc_list.GetSize(), 1)

        sb_proc_info = lldb.SBProcessInfo()
        self.assertTrue(proc_list.GetProcessInfoAtIndex(0, sb_proc_info))
        self.assertTrue(sb_proc_info.IsValid())

        self.assertEqual(sb_proc_info.GetName(), proc_info["name"])
        self.assertEqual(sb_proc_info.GetTriple(), proc_info["arch"])
        self.assertEqual(sb_proc_info.GetProcessID(), proc_info["pid"])
        self.assertEqual(sb_proc_info.GetParentProcessID(), proc_info["parent"])
        self.assertEqual(sb_proc_info.GetUserID(), proc_info["uid"])
        self.assertEqual(sb_proc_info.GetGroupID(), proc_info["gid"])
