"""
Test displayed value of a vector variable while doing watchpoint operations
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestValueOfVectorVariableTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_value_of_vector_variable_using_watchpoint_set(self):
        """Test verify displayed value of vector variable."""
        exe = self.getBuildArtifact("a.out")
        d = {"C_SOURCES": self.source, "EXE": exe}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.value_of_vector_variable_with_watchpoint_set()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Our simple source filename.
        self.source = "main.c"

    def value_of_vector_variable_with_watchpoint_set(self):
        """Test verify displayed value of vector variable"""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Set break to get a frame
        self.runCmd("b main")

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Value of a vector variable should be displayed correctly
        self.expect(
            "watchpoint set variable global_vector",
            WATCHPOINT_CREATED,
            substrs=["new value: (1, 2, 3, 4)"],
        )
