# TestSwiftConsumeOperator.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Check that we properly show variables at various points of the CFG while
stepping with the consume operator.
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import sys
import unittest2

def stderr_print(line):
    sys.stderr.write(line + "\n")

class TestSwiftConsumeOperatorType(TestBase):
    # Skip on aarch64 linux: rdar://91005071
    @skipIf(archs=['aarch64'], oslist=['linux'])
    @swiftTest
    def test_swift_consume_operator(self):
        """Check that we properly show variables at various points of the CFG while
        stepping with the consume operator.
        """
        self.build()

        self.target, self.process, self.thread, self.bkpt = \
            lldbutil.run_to_source_breakpoint(
                self, 'Set breakpoint', lldb.SBFileSpec('main.swift'))

        self.do_check_copyable_value_test()
        self.do_check_copyable_var_test()
        self.do_check_addressonly_value_test()
        self.do_check_addressonly_var_test()

        # argument simple tests
        self.do_check_copyable_value_arg_test()
        self.do_check_copyable_var_arg_test()
        self.do_check_addressonly_value_arg_test()
        self.do_check_addressonly_var_arg_test()

        # ccf is conditional control flow
        self.do_check_copyable_value_ccf_true()
        self.do_check_copyable_value_ccf_false()
        self.do_check_copyable_var_ccf_true_reinit_out_block()
        self.do_check_copyable_var_ccf_true_reinit_in_block()
        self.do_check_copyable_var_ccf_false_reinit_out_block()
        self.do_check_copyable_var_ccf_false_reinit_in_block()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.exec_name = "a.out"

    def get_var(self, name):
        frame = self.thread.frames[0]
        return frame.FindVariable(name)

    def do_check_copyable_value_test(self):
        # We haven't defined k yet.
        self.assertIsNone(self.get_var('k').value, "k initialized too early?!")

        # Go to break point 2. k should be valid.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to breakpoint 3. k should no longer be valid.
        self.process.Continue()
        self.assertIsNone(self.get_var('k').value, "K is live but was consumed?!")

        # Run so we hit the next breakpoint to jump to the next test's
        # breakpoint.
        self.process.Continue()

    def do_check_copyable_var_test(self):
        # We haven't defined k yet.
        self.assertIsNone(self.get_var('k').value, "k initialized too early?!")

        # Go to break point 2. k should be valid.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to breakpoint 3. We invalidated k
        self.process.Continue()
        self.assertIsNone(self.get_var('k').value, "K is live but was consumed?!")

        # Go to the last breakpoint and make sure that k is reinitialized
        # properly.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized")

        # Run so we hit the next breakpoint to go to the next test.
        self.process.Continue()

    def do_check_addressonly_value_test(self):
        # We haven't defined k yet.
        # Go to break point 2. k should be valid and m should not be. Since M is
        # a dbg.declare it is hard to test robustly that it is not initialized
        # so we don't do so. We have an additional llvm.dbg.addr test where we
        # move the other variable and show the correct behavior with
        # llvm.dbg.declare.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "var not initialized?!")

        # Go to breakpoint 3.
        self.process.Continue()
        self.assertEqual(self.get_var('k').unsigned, 0,
                        "dbg thinks k is live despite move?!")

        # Run so we hit the next breakpoint as part of the next test.
        self.process.Continue()

    def do_check_addressonly_var_test(self):
        # Go to break point 2. k should be valid.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to breakpoint 3. K was invalidated.
        self.process.Continue()
        self.assertIsNone(self.get_var('k').value, "K is live but was consumed?!")

        # Go to the last breakpoint and make sure that k is reinitialized
        # properly.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized")

        # Run so we hit the next breakpoint as part of the next test.
        self.process.Continue()

    def do_check_copyable_value_arg_test(self):
        # k is defined by the argument so it is valid.
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to break point 2. k should be valid.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to breakpoint 3. k should no longer be valid.
        self.process.Continue()
        #self.assertIsNone(self.get_var('k').value, "K is live but was consumed?!")

        # Run so we hit the next breakpoint to jump to the next test's
        # breakpoint.
        self.process.Continue()

    def do_check_copyable_var_arg_test(self):
        # k is already defined and is an argument.
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to break point 2. k should be valid.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to breakpoint 3. We invalidated k
        self.process.Continue()
        self.assertIsNone(self.get_var('k').value, "K is live but was consumed?!")

        # Go to the last breakpoint and make sure that k is reinitialized
        # properly.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized")

        # Run so we hit the next breakpoint to go to the next test.
        self.process.Continue()

    def do_check_addressonly_value_arg_test(self):
        # k is defined since it is an argument.
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to break point 2. k should be valid and m should not be. Since M is
        # a dbg.declare it is hard to test robustly that it is not initialized
        # so we don't do so. We have an additional llvm.dbg.addr test where we
        # move the other variable and show the correct behavior with
        # llvm.dbg.declare.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "var not initialized?!")

        # Go to breakpoint 3.
        self.process.Continue()
        self.assertEqual(self.get_var('k').unsigned, 0,
                        "dbg thinks k is live despite move?!")

        # Run so we hit the next breakpoint as part of the next test.
        self.process.Continue()

    def do_check_addressonly_var_arg_test(self):
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to break point 2. k should be valid.
        self.process.Continue()
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized?!")

        # Go to breakpoint 3. K was invalidated.
        self.process.Continue()
        self.assertIsNone(self.get_var('k').value, "K is live but was consumed?!")

        # Go to the last breakpoint and make sure that k is reinitialized
        # properly.
        self.process.Continue()
        # There is some sort of bug here. We should have the value here. For now
        # leave the next line commented out and validate we are not seeing the
        # value so we can detect change in behavior.
        self.assertGreater(self.get_var('k').unsigned, 0, "k not initialized")

        # Run so we hit the next breakpoint as part of the next test.
        self.process.Continue()

    def do_check_copyable_value_ccf_true(self):
        # Check at our start point that we do not have any state for k and
        # then continue to our next breakpoint.
        self.assertIsNone(self.get_var('k').value, "k should not have a value?!")
        self.process.Continue()

        # At this breakpoint, k should be defined since we are going to do
        # something with it.
        self.assertIsNotNone(self.get_var('k').value, "k should have a value?!")
        self.process.Continue()

        # At this breakpoint, we are now in the conditional control flow part of
        # the loop. Make sure that we can see k still.
        self.assertIsNotNone(self.get_var('k').value, "k should have a value?!")
        self.process.Continue()

        # Ok, we just performed the move. k should not be no longer initialized.
        self.assertIsNone(self.get_var('k').value, "k should not have a value?!")
        self.process.Continue()

        # Finally we left the conditional control flow part of the function. k
        # should still be None.
        self.assertIsNone(self.get_var('k').value, "k should not have a value!")

        # Run again so we go and run to the next test.
        self.process.Continue()

    def do_check_copyable_value_ccf_false(self):
        # Check at our start point that we do not have any state for k and
        # then continue to our next breakpoint.
        self.assertIsNone(self.get_var('k').value, "k should not have a value?!")
        self.process.Continue()

        # At this breakpoint, k should be defined since we are going to do
        # something with it.
        self.assertIsNotNone(self.get_var('k').value, "k should have a value?!")
        self.process.Continue()

        # At this breakpoint, we are now past the end of the conditional
        # statement. We know due to the move checking that k can not have any
        # uses that are reachable from the move. So it is safe to always not
        # provide the value here.
        self.assertIsNone(self.get_var('k').value, "k should have a value?!")

        # Run again so we go and run to the next test.
        self.process.Continue()

    def do_check_copyable_var_ccf_true_reinit_out_block(self):
        # At first we should not have a value for k.
        self.assertEqual(self.get_var('k').unsigned, 0, "k should be nullptr!")
        self.process.Continue()

        # Now we are in the conditional true block. K should be defined since we
        # are on the move itself.
        self.assertGreater(self.get_var('k').unsigned, 0, "k should not be nullptr!")
        self.process.Continue()

        # Now we have executed the move and we are about to run code using
        # m. Make sure that K is not available!
        self.assertEqual(self.get_var('k').unsigned, 0,
                         "k was already consumed! Should be nullptr")
        self.process.Continue()

        # We are now out of the conditional lexical block on the line of code
        # that redefines k. k should still be not available.
        self.assertEqual(self.get_var('k').unsigned, 0,
                         "k was already consumed! Should be nullptr")
        self.process.Continue()

        # Ok, we have now reinit k and are about to call a method on it. We
        # should be valid now.
        self.assertGreater(self.get_var('k').unsigned, 0,
                           "k should have be reinitialized?!")

        # Run again so we go and run to the next test.
        self.process.Continue()

    def do_check_copyable_var_ccf_true_reinit_in_block(self):
        # At first we should not have a value for k.
        self.assertEqual(self.get_var('k').unsigned, 0, "k should be nullptr!")
        self.process.Continue()

        # Now we are in the conditional true block. K should be defined since we
        # are on the move itself.
        self.assertGreater(self.get_var('k').unsigned, 0, "k should not be nullptr!")
        self.process.Continue()

        # Now we have executed the move and we are about to reinit k but have
        # not yet. Make sure we are not available!
        self.assertEqual(self.get_var('k').unsigned, 0,
                         "k was already consumed! Should be nullptr")
        self.process.Continue()

        # We are now still inside the conditional part of the code, but have
        # reinitialized k.
        self.assertGreater(self.get_var('k').unsigned, 0,
                           "k was reinit! Should be valid value!")
        self.process.Continue()

        # We now have left the conditional part of the function. k should still
        # be available.
        self.assertGreater(self.get_var('k').unsigned, 0,
                           "k should have be reinitialized?!")

        # Run again so we go and run to the next test.
        self.process.Continue()

    def do_check_copyable_var_ccf_false_reinit_out_block(self):
        # At first we should not have a value for k.
        self.assertEqual(self.get_var('k').unsigned, 0, "k should be nullptr!")
        self.process.Continue()

        # Now we are right above the beginning of the false check. k should
        # still be valid.
        self.assertGreater(self.get_var('k').unsigned, 0, "k should not be nullptr!")
        self.process.Continue()

        # Now we are after the conditional part of the code on the reinit
        # line. Since this is reachable from the move and we haven't reinit yet,
        # k should not be available.
        self.assertEqual(self.get_var('k').unsigned, 0,
                         "k was already consumed! Should be nullptr")
        self.process.Continue()

        # Ok, we have now reinit k and are about to call a method on it. We
        # should be valid now.
        self.assertGreater(self.get_var('k').unsigned, 0,
                           "k should have be reinitialized?!")

        # Run again so we go and run to the next test.
        self.process.Continue()

    def do_check_copyable_var_ccf_false_reinit_in_block(self):
        # At first we should not have a value for k.
        self.assertEqual(self.get_var('k').unsigned, 0, "k should be nullptr!")
        self.process.Continue()

        # Now we are on the doSomething above the false check. So k should be
        # valid.
        self.assertGreater(self.get_var('k').unsigned, 0, "k should not be nullptr!")
        self.process.Continue()

        # Now we are after the conditional scope. Since k was reinitialized in
        # the conditional scope, along all paths we are valid so k should
        # still be available.
        self.assertGreater(self.get_var('k').unsigned, 0,
                           "k should not be nullptr?!")

        # Run again so we go and run to the next test.
        self.process.Continue()
