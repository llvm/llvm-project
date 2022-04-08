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

    mydir = TestBase.compute_mydir(__file__)

    @swiftTest
    def test_swift_move_function_async(self):
        """Check that we properly show variables at various points of the CFG while
        stepping with the move function.
        """
        self.build()

        self.target, self.process, self.thread, self.bkpt = \
            lldbutil.run_to_source_breakpoint(
                self, 'Set breakpoint', lldb.SBFileSpec('main.swift'))

        # We setup a single breakpoint in copyable var test so we can disable it
        # after we hit it.
        self.do_setup_breakpoints()

        self.do_check_copyable_value_test()
        self.do_check_copyable_var_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        self.exec_name = "a.out"

    def get_var(self, name):
        frame = self.thread.frames[0]
        return frame.FindVariable(name)

    def do_setup_breakpoints(self):
        self.breakpoints = []
        pattern = 'Special breakpoint'
        brk = self.target.BreakpointCreateBySourceRegex(
            pattern, self.main_source_spec)
        self.assertGreater(brk.GetNumLocations(), 0, VALID_BREAKPOINT)
        self.breakpoints.append(brk)

    def do_check_copyable_value_test(self):
        # We haven't defined varK yet.
        varK = self.get_var('k')
        self.assertEqual(varK.unsigned, 0, "varK initialized too early?!")

        # Go to break point 2.1. k should be valid.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 2.2. k should still be valid. And we should be on the
        # other side of the force split.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 3. k should still be valid. We should be at the move
        # on the other side of the forceSplit.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # We are now at break point 4. We have moved k, it should be empty.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Finally, we are on the other side of the final force split. Make sure
        # the value still isn't available.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Run so we hit the next breakpoint to jump to the next test's
        # breakpoint.
        self.process.Continue()

    def do_check_copyable_var_test(self):
        # We haven't defined varK yet.
        varK = self.get_var('k')
        self.assertEqual(varK.unsigned, 0, "varK initialized too early?!")

        # Go to break point 2.1. k should be valid.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 2.2. k should still be valid. And we should be on the
        # other side of the force split.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Go to breakpoint 3. k should still be valid. We should be at the move
        # on the other side of the forceSplit.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # There is an instruction with the wrong debug location on Linux,
        # causing us to jump back to 'step backwards' to an earlier location
        # before we have run the move. So disable that breakpoint so when we
        # continue, we get to the appropriate location on Linux and other
        # platforms.
        #
        # TODO: Investigate why this is happening!
        self.runCmd('# Skipping bad loc by disabling earlier break point 6')
        self.runCmd('break dis 2')

        # We are now at break point 4. We have moved k, it should be empty.
        self.process.Continue()
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Now, we are on the other side of the final force split. Make sure
        # the value still isn't available.
        self.process.Continue()
        self.runCmd('# On other side of force split')
        varK = self.get_var('k')
        self.assertIsNone(varK.value, "K is live but was moved?!")

        # Finally, we have reinitialized k, look for k.
        self.process.Continue()
        self.runCmd('# After var reinit')
        varK = self.get_var('k')
        self.assertGreater(varK.unsigned, 0, "varK not initialized?!")

        # Run so we hit the next breakpoint to jump to the next test's
        # breakpoint.
        self.runCmd('# At end of routine!')
        self.process.Continue()
