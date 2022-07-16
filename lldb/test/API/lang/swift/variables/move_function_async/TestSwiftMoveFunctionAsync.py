# TestSwiftMoveFunctionAsync.py
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
stepping with the move function.
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

class TestSwiftMoveFunctionAsyncType(TestBase):

    @swiftTest
    def test_swift_move_function_async(self):
        """Check that we properly show variables at various points of the CFG while
        stepping with the move function.
        """
        self.build()

        self.breakpoints = []
        self.target, self.process, self.thread, bkpt = \
            lldbutil.run_to_source_breakpoint(
                self, 'Set breakpoint 00', lldb.SBFileSpec('main.swift'))
        self.breakpoints.append(bkpt)

        # We setup a single breakpoint in copyable var test so we can disable it
        # after we hit it.
        self.do_setup_breakpoints()

        self.do_check_copyable_value_test()
        self.do_check_copyable_var_test()

    def get_var(self, name):
        frame = self.thread.frames[0]
        return frame.FindVariable(name)

    def do_setup_breakpoints(self):
        for i in range(1, 11):
            bkpt = self.target.BreakpointCreateBySourceRegex(
                'Set breakpoint %02d'%i, lldb.SBFileSpec('main.swift'))
            self.assertGreater(bkpt.GetNumLocations(), 0, VALID_BREAKPOINT)
            self.breakpoints.append(bkpt)
        self.assertEqual(len(self.breakpoints), 11)

    def continue_to(self, bkpt_id):
        while self.process.is_alive and \
              not lldbutil.continue_to_breakpoint(self.process,
                                                  self.breakpoints[bkpt_id]):
            pass
        self.assertTrue(self.process.is_alive)

    def do_check_copyable_value_test(self):
        # We haven't defined varK yet.
        varK = self.get_var('k')
        self.assertEqual(varK.unsigned, 0, "varK initialized too early?!")

        # Go to break point 1.1. k should be valid.
        self.continue_to(1)
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint `1.2. k should still be valid. And we should be on the
        # other side of the force split.
        self.continue_to(1)
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 2. k should still be valid. We should be at the move
        # on the other side of the forceSplit.
        self.continue_to(2)
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # We are now at break point 3. We have moved k, it should be empty.
        self.continue_to(3)
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Finally, we are on the other side of the final force split. Make sure
        # the value still isn't available.
        self.continue_to(4)
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

    def do_check_copyable_var_test(self):
        # Run so we hit the next breakpoint to jump to the next test's
        # breakpoint.
        self.continue_to(5)
        # We haven't defined varK yet.
        varK = self.get_var('k')
        self.assertEqual(varK.unsigned, 0, "varK initialized too early?!")

        # Go to break point 6.1. k should be valid.
        self.continue_to(6)
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 6.2. k should still be valid. And we should be on the
        # other side of the force split.
        self.continue_to(6)
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 7. k should still be valid. We should be at the move
        # on the other side of the forceSplit.
        self.continue_to(7)
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # We are now at break point 8. We have moved k, it should be empty.
        self.continue_to(8)
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Now, we are on the other side of the final force split. Make sure
        # the value still isn't available.
        self.continue_to(9)
        self.runCmd('# On other side of force split')
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Finally, we have reinitialized k, look for k.
        self.continue_to(10)
        self.runCmd('# After var reinit')
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        self.runCmd('# At end of routine!')
