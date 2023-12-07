"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDataFormatterLibcxxQueue(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        self.namespace = "std"

    def check_variable(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())

        queue = self.namespace + "::queue"
        self.assertIn(queue, var.GetDisplayTypeName())
        self.assertEqual(var.GetNumChildren(), 5)
        for i in range(5):
            ch = var.GetChildAtIndex(i)
            self.assertTrue(ch.IsValid())
            self.assertEqual(ch.GetValueAsSigned(), i + 1)

    @expectedFailureAll(
        bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android"
    )
    @add_test_categories(["libc++"])
    def test(self):
        """Test that std::queue is displayed correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.check_variable("q1")
        self.check_variable("q2")
