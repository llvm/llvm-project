"""
Test that we do not attempt to make a dynamic type for a 'const char*'
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCaseCharStarDynType(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

    def test_charstar_dyntype(self):
        """Test that we do not attempt to make a dynamic type for a 'const char*'"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here.", lldb.SBFileSpec("main.m")
        )

        # check that we correctly see the const char*, even with dynamic types
        # on
        self.expect("frame variable -raw-output my_string", substrs=["const char *"])
        self.expect(
            "frame variable my_string --raw-output --dynamic-type run-target",
            substrs=["const char *"],
        )
        # check that expr also gets it right
        self.expect("expr -R -- my_string", substrs=["const char *"])
        self.expect("expr -R -d run -- my_string", substrs=["const char *"])
        # but check that we get the real Foolie as such
        self.expect("frame variable my_foolie", substrs=["FoolMeOnce *"])
        self.expect(
            "frame variable my_foolie --dynamic-type run-target",
            substrs=["FoolMeOnce *"],
        )
        # check that expr also gets it right
        self.expect("expr my_foolie", substrs=["FoolMeOnce *"])
        self.expect("expr -d run -- my_foolie", substrs=["FoolMeOnce *"])
        # now check that assigning a true string does not break anything
        self.runCmd("next")
        # check that we correctly see the const char*, even with dynamic types
        # on
        self.expect("frame variable my_string", substrs=["const char *"])
        self.expect(
            "frame variable my_string --dynamic-type run-target",
            substrs=["const char *"],
        )
        # check that expr also gets it right
        self.expect("expr my_string", substrs=["const char *"])
        self.expect("expr -d run -- my_string", substrs=["const char *"])
        # but check that we get the real Foolie as such
        self.expect("frame variable my_foolie", substrs=["FoolMeOnce *"])
        self.expect(
            "frame variable my_foolie --dynamic-type run-target",
            substrs=["FoolMeOnce *"],
        )
        # check that expr also gets it right
        self.expect("expr my_foolie", substrs=["FoolMeOnce *"])
        self.expect("expr -d run -- my_foolie", substrs=["FoolMeOnce *"])
