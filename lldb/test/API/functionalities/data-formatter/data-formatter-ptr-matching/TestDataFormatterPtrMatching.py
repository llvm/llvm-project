"""
Test lldb data formatter subsystem.
"""

import os
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PtrMatchingDataFormatterTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number("main.cpp", "// Set break point at this line.")

    def test_summary_with_command(self):
        """Test "type summary add" command line option "--pointer-match-depth"."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # By default, --pointer-match-depth is 1.
        self.runCmd('type summary add --cascade true -s "MyInt" "Int"')
        self.expect(
            "frame variable",
            patterns=[
                r".* i = MyInt\n",
                r".* i_p = 0x.* MyInt\n",
                r".* i_pp = 0x[0-9a-f]+\n",
                r".* i_ppp = 0x[0-9a-f]+\n",
                r".* f = MyInt\n",
                r".* f_p = 0x[0-9a-f]+ MyInt\n",
                r".* f_pp = 0x[0-9a-f]+\n",
                r".* f_ppp = 0x[0-9a-f]+\n",
                r".* fp = 0x[0-9a-f]+ MyInt\n",
                r".* fp_p = 0x[0-9a-f]+\n",
                r".* fp_pp = 0x[0-9a-f]+\n",
                r".* b = MyInt\n",
                r".* b_p = 0x[0-9a-f]+ MyInt\n",
                r".* b_pp = 0x[0-9a-f]+\n",
                r".* bp = 0x[0-9a-f]+ MyInt\n",
                r".* bp_p = 0x[0-9a-f]+\n",
                r".* bp_pp = 0x[0-9a-f]+\n",
            ],
        )

        self.runCmd('type summary delete "Int"')
        self.runCmd(
            'type summary add --cascade true --pointer-match-depth 2 -s "MyInt" "Int"'
        )
        self.expect(
            "frame variable",
            patterns=[
                r".* i = MyInt\n",
                r".* i_p = 0x.* MyInt\n",
                r".* i_pp = 0x[0-9a-f]+ MyInt\n",
                r".* i_ppp = 0x[0-9a-f]+\n",
                r".* f = MyInt\n",
                r".* f_p = 0x[0-9a-f]+ MyInt\n",
                r".* f_pp = 0x[0-9a-f]+ MyInt\n",
                r".* f_ppp = 0x[0-9a-f]+\n",
                r".* fp = 0x[0-9a-f]+ MyInt\n",
                r".* fp_p = 0x[0-9a-f]+ MyInt\n",
                r".* fp_pp = 0x[0-9a-f]+\n",
                r".* b = MyInt\n",
                r".* b_p = 0x[0-9a-f]+ MyInt\n",
                r".* b_pp = 0x[0-9a-f]+ MyInt\n",
                r".* bp = 0x[0-9a-f]+ MyInt\n",
                r".* bp_p = 0x[0-9a-f]+ MyInt\n",
                r".* bp_pp = 0x[0-9a-f]+\n",
            ],
        )

        self.runCmd('type summary delete "Int"')
        self.runCmd(
            'type summary add --cascade true --pointer-match-depth 2 -s "MyFoo" "Foo"'
        )
        self.expect(
            "frame variable",
            patterns=[
                r".* f = MyFoo\n",
                r".* f_p = 0x[0-9a-f]+ MyFoo\n",
                r".* f_pp = 0x[0-9a-f]+ MyFoo\n",
                r".* f_ppp = 0x[0-9a-f]+\n",
                r".* fp = 0x[0-9a-f]+\n",
                r".* fp_p = 0x[0-9a-f]+\n",
                r".* fp_pp = 0x[0-9a-f]+\n",
                r".* b = MyFoo\n",
                r".* b_p = 0x[0-9a-f]+ MyFoo\n",
                r".* b_pp = 0x[0-9a-f]+ MyFoo\n",
                r".* bp = 0x[0-9a-f]+ MyFoo\n",
                r".* bp_p = 0x[0-9a-f]+ MyFoo\n",
                r".* bp_pp = 0x[0-9a-f]+\n",
            ],
        )
