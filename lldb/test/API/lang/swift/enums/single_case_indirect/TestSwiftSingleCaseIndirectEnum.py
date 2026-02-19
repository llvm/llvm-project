import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftSingleCaseIndirectEnum(TestBase):
    @swiftTest
    def test(self):
        """Test single-case indirect enum projection"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        frame = thread.frames[0]

        # Single-case indirect enum with a struct payload.
        single_struct = frame.FindVariable("single_struct")
        self.assertEqual(single_struct.GetNumChildren(), 2)
        lldbutil.check_variable(
            self, single_struct.GetChildMemberWithName("x"), False, value="42"
        )
        lldbutil.check_variable(
            self, single_struct.GetChildMemberWithName("y"), False,
            summary='"hello"',
        )

        # Single-case indirect enum with a class payload.
        single_class = frame.FindVariable("single_class")
        self.assertEqual(single_class.GetNumChildren(), 1)
        lldbutil.check_variable(
            self, single_class.GetChildMemberWithName("v"), False, value="100"
        )

        # Single-case indirect enum with a tuple payload.
        single_tuple = frame.FindVariable("single_tuple")
        self.assertEqual(single_tuple.GetNumChildren(), 2)
        lldbutil.check_variable(
            self, single_tuple.GetChildAtIndex(0), False, value="7"
        )
        lldbutil.check_variable(
            self, single_tuple.GetChildAtIndex(1), False, summary='"world"'
        )

        # Single-case indirect enum with a large struct payload.
        single_large = frame.FindVariable("single_large")
        self.assertEqual(single_large.GetNumChildren(), 4)
        lldbutil.check_variable(
            self, single_large.GetChildMemberWithName("a"), False, value="1"
        )
        lldbutil.check_variable(
            self, single_large.GetChildMemberWithName("b"), False, value="2"
        )
        lldbutil.check_variable(
            self, single_large.GetChildMemberWithName("c"), False, value="3"
        )
        lldbutil.check_variable(
            self, single_large.GetChildMemberWithName("d"), False,
            summary='"big"',
        )

        # Single-case indirect enum with a scalar payload.
        single_int = frame.FindVariable("single_int")
        self.assertEqual(single_int.GetNumChildren(), 1)
        lldbutil.check_variable(
            self, single_int.GetChildAtIndex(0), False, value="99"
        )

        # Recursive single-case indirect enum (linked list).
        list_var = frame.FindVariable("list")
        self.assertEqual(list_var.GetNumChildren(), 2)
        lldbutil.check_variable(
            self, list_var.GetChildAtIndex(0), False, value="10"
        )
        # Second child is Optional<List<Int>> wrapping the next node.
        node2 = list_var.GetChildAtIndex(1)
        self.assertEqual(node2.GetNumChildren(), 2)
        lldbutil.check_variable(
            self, node2.GetChildAtIndex(0), False, value="20"
        )
        node3 = node2.GetChildAtIndex(1)
        self.assertEqual(node3.GetNumChildren(), 2)
        lldbutil.check_variable(
            self, node3.GetChildAtIndex(0), False, value="30"
        )
        # Tail is nil.
        tail = node3.GetChildAtIndex(1)
        lldbutil.check_variable(self, tail, False, num_children=0,
                                summary="nil")
