"""
Test lldb data formatter subsystem.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDataFormatterStdQueue(TestBase):
    SHARED_BUILD_TESTCASE = False

    def setUp(self):
        TestBase.setUp(self)
        self.namespace = "std"

    def check_sequence(self, name, type_name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid(), name)
        self.assertIn(self.namespace + "::" + type_name, var.GetDisplayTypeName())
        self.assertEqual(var.GetNumChildren(), 5, name)
        for i in range(5):
            ch = var.GetChildAtIndex(i)
            self.assertTrue(ch.IsValid(), f"{name}[{i}]")
            self.assertEqual(ch.GetValueAsSigned(), i + 1, f"{name}[{i}]")

    def check_priority_queue(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid(), name)
        self.assertIn(self.namespace + "::priority_queue", var.GetDisplayTypeName())
        self.assertEqual(var.GetNumChildren(), 5, name)
        values = sorted(var.GetChildAtIndex(i).GetValueAsSigned() for i in range(5))
        self.assertEqual(values, [1, 2, 3, 4, 5])

    def do_test_queues(self):
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )
        self.check_sequence("q1", "queue")
        self.check_sequence("q2", "queue")

    def do_test_adaptors(self):
        self.do_test_queues()
        self.check_sequence("s1", "stack")
        self.check_sequence("s2", "stack")
        self.check_priority_queue("pq")

    @expectedFailureAll(
        bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android"
    )
    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_adaptors()

    @expectedFailureAll(
        bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android"
    )
    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_queues()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        self.do_test_adaptors()
