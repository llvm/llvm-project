"""
Test situations where we don't have a definition for a type, but we have (some)
of its member functions.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCppIncompleteTypeMembers(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
                lldb.SBFileSpec("f.cpp"))

        # Sanity check that we really have to debug info for this type.
        this = self.expect_var_path("this", type="A *")
        self.assertEquals(this.GetType().GetPointeeType().GetNumberOfFields(),
                0, str(this))

        self.expect_var_path("af.x", value='42')

        lldbutil.run_break_set_by_source_regexp(self, "// break here",
                extra_options="-f g.cpp")
        self.runCmd("continue")

        self.expect_var_path("ag.a", value='47')
