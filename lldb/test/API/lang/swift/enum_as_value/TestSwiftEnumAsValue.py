import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftAnyType(lldbtest.TestBase):

    @swiftTest
    def test_any_type(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'Set breakpoint here', lldb.SBFileSpec('main.swift'))

        frame = thread.frames[0]
        var_e = frame.FindVariable("e")
        # We don't have an encoding for enums.
        self.assertEqual(var_e.GetValueAsSigned(0xdead), 0xdead)
        self.assertEqual(var_e.GetValue(), 'B')
