"""Test custom import command to import files by path."""

import os, sys, time
import unittest2
import lldb
from lldbtest import *

class ImportTestCase(TestBase):

    mydir = os.path.join("functionalities", "command_script", "import")

    @python_api_test
    @skipIfLinux # causes buildbot failures, skip until we can investigate it
    def test_import_command(self):
        """Import some Python scripts by path and test them"""
        self.run_test()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def run_test(self):
        """Import some Python scripts by path and test them."""

        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('command script delete foo2cmd', check=False)
            self.runCmd('command script delete foocmd', check=False)
            self.runCmd('command script delete foobarcmd', check=False)
            self.runCmd('command script delete barcmd', check=False)
            self.runCmd('command script delete barothercmd', check=False)
            self.runCmd('command script delete TPcommandA', check=False)
            self.runCmd('command script delete TPcommandB', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("command script import ./foo/foo.py --allow-reload")
        self.runCmd("command script import ./foo/foo2.py --allow-reload")
        self.runCmd("command script import ./foo/bar/foobar.py --allow-reload")
        self.runCmd("command script import ./bar/bar.py --allow-reload")

        self.expect("command script import ./nosuchfile.py",
                error=True, startstr='error: module importing failed')
        self.expect("command script import ./nosuchfolder/",
                error=True, startstr='error: module importing failed')
        self.expect("command script import ./foo/foo.py", error=False)

        self.runCmd("command script import --allow-reload ./thepackage")
        self.expect("TPcommandA",substrs=["hello world A"])
        self.expect("TPcommandB",substrs=["hello world B"])

        self.runCmd("script import dummymodule")
        self.expect("command script import ./dummymodule.py", error=False)
        self.expect("command script import --allow-reload ./dummymodule.py", error=False)

        self.runCmd("command script add -f foo.foo_function foocmd")
        self.runCmd("command script add -f foobar.foo_function foobarcmd")
        self.runCmd("command script add -f bar.bar_function barcmd")
        self.expect("foocmd hello",
                substrs = ['foo says', 'hello'])
        self.expect("foo2cmd hello",
                substrs = ['foo2 says', 'hello'])
        self.expect("barcmd hello",
                substrs = ['barutil says', 'bar told me', 'hello'])
        self.expect("barothercmd hello",
                substrs = ['barutil says', 'bar told me', 'hello'])
        self.expect("foobarcmd hello",
                substrs = ['foobar says', 'hello'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
