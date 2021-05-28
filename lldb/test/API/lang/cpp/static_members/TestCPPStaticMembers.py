"""
Tests that C++ member and static variables have correct layout and scope.
"""



import unittest2
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    # We fail to lookup static members on Windows.
    @expectedFailureAll(oslist=["windows"])
    def test_access_from_main(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// stop in main", lldb.SBFileSpec("main.cpp"))

        self.expect_expr("my_a.m_a", result_type="short", result_value="1")
        self.expect_expr("my_a.s_b", result_type="long", result_value="2")
        self.expect_expr("my_a.s_c", result_type="int", result_value="3")

    # We fail to lookup static members on Windows.
    @expectedFailureAll(oslist=["windows"])
    def test_access_from_member_function(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// stop in member function", lldb.SBFileSpec("main.cpp"))
        self.expect_expr("m_a", result_type="short", result_value="1")
        self.expect_expr("s_b", result_type="long", result_value="2")
        self.expect_expr("s_c", result_type="int", result_value="3")

    # Currently lookups find variables that are in any scope.
    @expectedFailureAll()
    def test_access_without_scope(self):
        self.build()
        self.createTestTarget()
        self.expect("expression s_c", error=True,
                    startstr="error: use of undeclared identifier 's_d'")
