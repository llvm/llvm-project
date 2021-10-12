"""
Test user added container commands
"""


import sys
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestCmdContainer(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def test_container_add(self):
        self.container_add()

    def check_command_tree_exists(self):
        """This makes sure we can still run the command tree we added."""
        self.runCmd("test-multi")
        self.runCmd("test-multi test-multi-sub")
        self.runCmd("test-multi test-multi-sub welcome")
        
    def container_add(self):
        # Make sure we can't overwrite built-in commands:
        self.expect("command container add process", "Can't replace builtin container command",
                    substrs=["can't replace builtin command"], error=True)
        self.expect("command container add process non_such_subcommand", "Can't add to built-in subcommand", 
                    substrs=["Path component: 'process' is not a user command"], error=True)
        self.expect("command container add process launch", "Can't replace builtin subcommand", 
                    substrs=["Path component: 'process' is not a user command"], error=True)

        # Now lets make a container command:
        self.runCmd("command container add -h 'A test container command' test-multi")
        # Make sure the help works:
        self.expect("help test-multi", "Help works for top-level multi",
                    substrs=["A test container command"])
        # Add a subcommand:
        self.runCmd("command container add -h 'A test container sub-command' test-multi test-multi-sub")
        # Make sure the help works:
        self.expect("help test-multi", "Help shows sub-multi",
                    substrs=["A test container command", "test-multi-sub -- A test container sub-command"])
        self.expect("help test-multi test-multi-sub", "Help shows sub-multi",
                    substrs=["A test container sub-command"])

        # Now add a script based command to the container command:
        self.runCmd("command script import welcome.py")
        self.runCmd("command script add -c welcome.WelcomeCommand test-multi test-multi-sub welcome")
        # Make sure the help still works:
        self.expect("help test-multi test-multi-sub", "Listing subcommands works",
                    substrs=["A test container sub-command", "welcome -- Just a docstring for Welcome"])
        self.expect("help test-multi test-multi-sub welcome", "Subcommand help works",
                    substrs=["Just a docstring for Welcome"])
        # And make sure it actually works:
        self.expect("test-multi test-multi-sub welcome friend", "Test command works",
                    substrs=["Hello friend, welcome to LLDB"])

        # Make sure overwriting works, first the leaf command:
        # We should not be able to remove extant commands by default:
        self.expect("command script add -c welcome.WelcomeCommand2 test-multi test-multi-sub welcome",
                    "overwrite command w/o -o",
                    substrs=["cannot add command: sub-command already exists"], error=True)
        # But we can with the -o option:
        self.runCmd("command script add -c welcome.WelcomeCommand2 -o test-multi test-multi-sub welcome")
        # Make sure we really did overwrite:
        self.expect("test-multi test-multi-sub welcome friend", "Used the new command class",
                    substrs=["Hello friend, welcome again to LLDB"])

        self.expect("apropos welcome", "welcome should show up in apropos", substrs=["Just a docstring for the second Welcome"])
        
        # Make sure we give good errors when the input is wrong:
        self.expect("command script delete test-mult test-multi-sub welcome", "Delete script command - wrong first path component",
                    substrs=["'test-mult' not found"], error=True)
        
        self.expect("command script delete test-multi test-multi-su welcome", "Delete script command - wrong second path component",
                    substrs=["'test-multi-su' not found"], error=True)
        self.check_command_tree_exists()
        
        self.expect("command script delete test-multi test-multi-sub welcom", "Delete script command - wrong leaf component",
                    substrs=["'welcom' not found"], error=True)
        self.check_command_tree_exists()
        
        self.expect("command script delete test-multi test-multi-sub", "Delete script command - no leaf component",
                    substrs=["subcommand 'test-multi-sub' is not a user command"], error=True)
        self.check_command_tree_exists()

        # You can't use command script delete to delete container commands:
        self.expect("command script delete test-multi", "Delete script - can't delete container",
                    substrs=["command 'test-multi' is a multi-word command."], error=True)
        self.expect("command script delete test-multi test-multi-sub", "Delete script - can't delete container",
                    substrs=["subcommand 'test-multi-sub' is not a user command"], error = True)

        # You can't use command container delete to delete scripted commands:
        self.expect("command container delete test-multi test-multi-sub welcome", "command container can't delete user commands",
                    substrs=["subcommand 'welcome' is not a container command"], error = True)
        
        # Also make sure you can't alias on top of container commands:
        self.expect("command alias test-multi process launch", "Tried to alias on top of a container command",
                    substrs=["'test-multi' is a user container command and cannot be overwritten."], error=True)
        self.check_command_tree_exists()

        # Also assert that we can't delete builtin commands:
        self.expect("command script delete process launch", "Delete builtin command fails", substrs=["command 'process' is not a user command"], error=True)
        # Now let's do the version that works
        self.expect("command script delete test-multi test-multi-sub welcome", "Delete script command by path", substrs=["Deleted command: test-multi test-multi-sub welcome"])

        # Now overwrite the sub-command, it should end up empty:
        self.runCmd("command container add -h 'A different help string' -o test-multi test-multi-sub")
        # welcome should be gone:
        self.expect("test-multi test-multi-sub welcome friend", "did remove subcommand",
                    substrs=["'test-multi-sub' does not have any subcommands."], error=True)
        # We should have the new help:
        self.expect("help test-multi test-multi-sub", "help changed",
                    substrs=["A different help string"])

        # Now try deleting commands.
        self.runCmd("command container delete test-multi test-multi-sub")
        self.expect("test-multi test-multi-sub", "Command is not active", error=True,
                    substrs = ["'test-multi' does not have any subcommands"])
        self.expect("help test-multi", matching=False, substrs=["test-multi-sub"])

                    
        # Next the root command:
        self.runCmd("command container delete test-multi")
        self.expect("test-multi", "Root command gone", substrs=["'test-multi' is not a valid command."], error=True)
