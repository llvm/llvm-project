"""
Test python scripted process in lldb
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class PlatformProcessCrashInfoTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.source = "main.c"

    def tearDown(self):
        TestBase.tearDown(self)

    def test_python_plugin_package(self):
        """Test that the lldb python module has a `plugins.scripted_process`
        package."""
        self.expect('script import lldb.plugins',
                    substrs=["ModuleNotFoundError"], matching=False)

        self.expect('script dir(lldb.plugins)',
                    substrs=["scripted_process"])

        self.expect('script import lldb.plugins.scripted_process',
                    substrs=["ModuleNotFoundError"], matching=False)

        self.expect('script dir(lldb.plugins.scripted_process)',
                    substrs=["ScriptedProcess"])

        self.expect('script from lldb.plugins.scripted_process import ScriptedProcess',
                    substrs=["ImportError"], matching=False)

        self.expect('script dir(ScriptedProcess)',
                    substrs=["launch"])

