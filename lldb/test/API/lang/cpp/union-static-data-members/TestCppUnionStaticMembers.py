"""
Tests that frame variable and expr work for
C++ unions and their static data members.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class CppUnionStaticMembersTestCase(TestBase):
    def test(self):
        """Tests that frame variable and expr work
           for union static data members"""
        self.build()

        (target, process, main_thread, _) = lldbutil.run_to_source_breakpoint(
            self, "return 0", lldb.SBFileSpec("main.cpp")
        )                                                                     

        self.expect("frame variable foo", substrs=["val = 42"])
        self.expect("frame variable bar", substrs=["val = 137"])

        self.expect_expr("foo", result_type="Foo", result_children=[ValueCheck(
                name="val", value="42"
            )])
        self.expect_expr("bar", result_type="Bar", result_children=[ValueCheck(
                name="val", value="137"
            )])

        self.expect_expr("Foo::sVal1", result_type="const int", result_value="-42")
        self.expect_expr("Foo::sVal2", result_type="Foo", result_children=[ValueCheck(
                name="val", value="42"
            )])

    @expectedFailureAll
    def test_union_in_anon_namespace(self):
        """Tests that frame variable and expr work
           for union static data members in anonymous
           namespaces"""
        self.expect_expr("Bar::sVal1", result_type="const int", result_value="-137")
        self.expect_expr("Bar::sVal2", result_type="Bar", result_children=[ValueCheck(
                name="val", value="137"
            )])
