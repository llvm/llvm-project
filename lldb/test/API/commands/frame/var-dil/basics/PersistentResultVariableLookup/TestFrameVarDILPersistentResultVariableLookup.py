"""
Make sure accessing persistent/result variables works using DIL parser/evaluator.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILPersistentResultVariableLookup(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    def test_frame_var(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        # Establish persistent variables.

        # Establish a persistent variable with integer type.
        self.expect(
            "dwim-print --persistent-result true -- foo + 5", startstr="(int) $0 = "
        )
        # Establish a persistent variable using dwim-print with derived type (using C specification terminology).
        self.expect(
            "dwim-print --persistent-result true -- hsmt",
            startstr="(HasMembersT) $1 = ",
        )
        # Establish a persistent variable (using dwim-print) as pointer to variable
        # with derived type (using C specification terminology).
        self.expect(
            "dwim-print --persistent-result true -- &hsmt",
            startstr="(HasMembersT *) $2 = ",
        )
        # Establish result variable using expression.
        self.expect(
            "expression foo",
            startstr="(int) $3 = 1",
        )
        # Establish result variables with user-defined names.
        self.runCmd(
            "expression int *$foop = &foo",
        )
        self.runCmd(
            "expression HasMembersT *$hsmtp = &hsmt",
        )

        # Test that accessing those persistent/result variables yields the proper values.

        # Check to make sure that `dwim-print`'s note indicates the proper path through the code was taken.
        self.runCmd("settings set dwim-print-verbosity full")
        # Check that `dwim-print` uses DIL.
        self.expect("dwim-print $0", startstr="note: ran `frame variable $0`")
        self.expect(
            "dwim-print $1.doublem", startstr="note: ran `frame variable $1.doublem`"
        )

        # Check that `dwim-print` looks up the name directly.
        # (requires temporarily disabling DIL)
        self.runCmd("settings set target.experimental.use-DIL false")
        self.expect("dwim-print $3", startstr="(int) 1")
        self.runCmd("settings set target.experimental.use-DIL true")

        # Check that `dwim-print` uses `expression` command.
        self.expect(
            "dwim-print $2->doublem", startstr="note: ran `expression $2->doublem`"
        )

        self.runCmd("settings set dwim-print-verbosity none")

        # Check simple persistent variable accesses.
        self.expect_var_path("$0", type="int", value="6")
        self.expect_var_path(
            "$1",
            type="HasMembersT",
            children=[
                ValueCheck(name="intm", value="1", type="int"),
                ValueCheck(name="doublem", value="2", type="double"),
                ValueCheck(
                    name="nestedm",
                    type="NestedT",
                    children=[ValueCheck(name="charm", type="char", value="'c'")],
                ),
            ],
        )
        self.expect_var_path(
            "$2",
            type="HasMembersT *",
            children=[
                ValueCheck(name="intm", value="1", type="int"),
                ValueCheck(name="doublem", value="2", type="double"),
                ValueCheck(
                    name="nestedm",
                    type="NestedT",
                    children=[ValueCheck(name="charm", type="char", value="'c'")],
                ),
            ],
        )
        self.expect_var_path("$3", type="int", value="1")

        # Check that accessing fields of persistent variables works.
        self.expect_var_path("$1.intm", type="int", value="1")
        self.expect_var_path("$1.nestedm.charm", type="char", value="'c'")
        self.expect_var_path("$1.intm + $0", type="int", value="7")

        # Check that types work correctly when adding an int and a double.
        self.expect_var_path("$1.intm + $1.doublem", type="double", value="3")

        self.expect_var_path("*$foop", type="int", value="1")
        self.expect_var_path("(*$hsmtp).doublem", type="double", value="2")

        # Step past statements that update variable values to which persistent
        # variables refer.
        lldbutil.continue_to_source_breakpoint(
            self, process, "Set a second breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        # Make sure that the value accessed through the pointer in persistent variables are updated.
        self.expect_var_path("*$foop", type="int", value="2")
        self.expect_var_path("(*$hsmtp).doublem", type="double", value="3")
