# TestSwiftStepping.py
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
Tests simple swift stepping
"""
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import platform
import unittest2

class TestSwiftStepping(lldbtest.TestBase):

    @swiftTest
    @skipIfLinux
    def test_swift_stepping(self):
        """Tests that we can step reliably in swift code."""
        self.build()
        self.do_test()

    def setUp(self):
        lldbtest.TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)
        # If you are running against a debug swift you are going to
        # end up stepping into the stdlib and that will make stepping
        # tests impossible to write.  So avoid that.

        if platform.system() == 'Darwin':
            lib_name = "libswiftCore.dylib"
        else:
            lib_name = "libswiftCore.so"

        self.dbg.HandleCommand(
            "settings set "
            "target.process.thread.step-avoid-libraries {}".format(lib_name))

    def correct_stop_reason(self, thread):
        # We are always just stepping, so that should be the thread stop reason:
        stop_reason = thread.GetStopReason()
        self.assertEqual(stop_reason, lldb.eStopReasonPlanComplete)

    def hit_correct_line(self, thread, pattern, fail_if_wrong=True):
        # print "Check if we got to: ", pattern
        self.correct_stop_reason(thread)
        target_line = lldbtest.line_number(self.main_source, pattern)
        self.assertTrue(
            target_line != 0,
            "Could not find source pattern " + pattern)
        cur_line = thread.frames[0].GetLineEntry().GetLine()
        hit_line = cur_line == target_line
        if fail_if_wrong:
            self.assertTrue(
                hit_line,
                "Stepped to line %d instead of expected %d "
                "with pattern '%s'." % (cur_line, target_line, pattern))
        return hit_line

    def hit_correct_function(self, thread, pattern):
        # print "Check if we got to: ", pattern
        self.correct_stop_reason(thread)
        name = thread.frames[0].GetFunctionName()
        line_entry = thread.frames[0].GetLineEntry()
        desc = lldb.SBStream()
        if line_entry.IsValid():
            line_entry.GetDescription(desc)
        else:
            desc.Print(name)
        self.assertTrue(
            pattern in name,
            "Got to '%s' not the expected function '%s'." % (desc.GetData(), pattern))

    def do_test(self):
        """Tests that we can step reliably in swift code."""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, lldbtest.VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Stop here first in main', self.main_source_spec)
        self.assertTrue(
            breakpoint.GetNumLocations() > 0, lldbtest.VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, lldbtest.PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        thread = threads[0]
        frame = thread.frames[0]
        self.assertTrue(frame, "Frame 0 is valid.")

        # We are before using an int to set an enum value (which has a
        # toSomeValues function).  So step into should go into the
        # toSomeValues function.

        thread.StepInto()
        self.hit_correct_function(thread, "toSomeValues")

        thread.StepOut()
        self.hit_correct_function(thread, "main")

        thread.StepOver()
        thread.StepOver()

        # That should take me to the equals operator.
        self.hit_correct_line(thread, "Stop here to get into equality")

        thread.StepInto()
        self.hit_correct_function(thread, "==")

        thread.StepOut()
        thread.StepOver()

        self.hit_correct_line(thread, "Step over the if should get here")

        thread.StepOver()
        self.hit_correct_line(thread, "Step over the print should get here.")

        thread.StepOver()
        self.hit_correct_line(thread, "Stop here to step into B constructor.")

        thread.StepInto()
        prologue_at_first_line = self.hit_correct_line(
            thread, "At the first line of B constructor.", False)
        if prologue_at_first_line:
            thread.StepOver()

        self.hit_correct_line(
            thread, "In the B constructor about to call super.")

        # Make sure we can follow super into the parent constructor.
        thread.StepInto()

        prologue_at_first_line = self.hit_correct_line(
            thread, "Here is the A constructor.", False)
        if prologue_at_first_line:
            thread.StepOver()

        self.hit_correct_line(thread, "A init: x assignment.")

        thread.StepOver()
        self.hit_correct_line(thread, "A init: y assignment.")

        thread.StepOver()
        self.hit_correct_line(thread, "A init: end of function.")

        # Now two step outs should take us back where we started from:
        thread.StepOut()
        thread.StepOut()
        name = thread.frames[0].GetFunctionName()
        if "allocating_init" in name:
            thread.StepOut()
        self.hit_correct_line(thread, "Stop here to step into B constructor.")

        thread.StepOver()
        self.hit_correct_line(
            thread, "Stop here to step into call_overridden.")

        # Make sure we get into the overridden call as expected:
        thread.StepInto()
        self.hit_correct_function(thread, "call_overridden")
        stopped_at_func_defn = self.hit_correct_line(
            thread, "call_overridden func def", False)
        if stopped_at_func_defn:
            thread.StepOver()
        thread.StepInto()
        self.hit_correct_function(thread, "ClassB.do_something")

        # Now step out twice to get back to main:
        thread.StepOut()
        self.hit_correct_function(thread, "call_overridden")
        thread.StepOut()
        self.hit_correct_line(
            thread, "Stop here to step into call_overridden.")

        # Two steps should get us to the switch:
        thread.StepOver()
        self.hit_correct_line(thread, "At point initializer.")
        thread.StepOver()
        self.hit_correct_line (thread, "At the beginning of the switch.")

        thread.StepOver()
        stopped_at_case = self.hit_correct_line(
            thread, "case (let x, let y) where", False)
        if stopped_at_case:
            thread.StepOver()

        self.hit_correct_line(thread, "First case with a where statement.")
        thread.StepOver()
        self.hit_correct_line(thread, "Second case with a where statement")

        thread.StepInto()
        self.hit_correct_line(
            thread, "return_same gets called in both where statements")

        thread.StepOut()
        self.hit_correct_line(thread, "Second case with a where statement")

        thread.StepOver()
        self.hit_correct_line(
            thread, "print in second case with where statement.")

        #
        # For some reason, we don't step from the body of the case to
        # the end of the switch, but back to the case: statement, and
        # then directly out of the switch.
        #
        thread.StepOver()
        steps_back_to_case = self.hit_correct_line(
            thread,
            "Sometimes the line table steps to here "
            "after the body of the case.", False)
        if steps_back_to_case:
            self.fail(
                "Stepping past a taken body of a case statement should not "
                "step back to the case statement.")

        if self.hit_correct_line(
                thread,
                "This is the end of the switch statement", False):
            thread.StepOver()
        elif not self.hit_correct_line(
                thread, "Make a version of P that conforms directly", False):
            self.fail(
                "Stepping past the body of the case didn't stop "
                "where expected.")

        self.hit_correct_line(
            thread, "Make a version of P that conforms directly")

        # FIXME: Stepping into constructors is kind of a mess,
        # I'm going to just step over for these tests.
        thread.StepOver()
        self.hit_correct_line(thread, "direct.protocol_func(10)")
        # This tests stepping in through the protocol thunk:
        thread.StepInto()

        stop_at_prologue = self.hit_correct_line(
            thread,
            "We stopped at the protocol_func declaration instead.",
            False)
        if stop_at_prologue:
            thread.StepOver()
        self.hit_correct_line(
            thread, "This is where we will stop in the protocol dispatch")

        # Step out of the protocol functions, one step should get us
        # past any dispatch thunk.
        thread.StepOut()

        # Finish may not get us past the line we were on, since it
        # stops immediately on returning
        # to the caller frame.  If so we will need to step again:
        stop_in_caller_line = self.hit_correct_line(
            thread, "direct.protocol_func(10)", False)
        if stop_in_caller_line:
            thread.StepOver()
        self.hit_correct_line(
            thread, "Make a version of P that conforms through a subclass")

        # This steps into another protocol dispatch.
        thread.StepOver()
        thread.StepInto()
        stop_at_prologue = self.hit_correct_line(
            thread,
            "We stopped at the protocol_func declaration instead.",
            False)
        if stop_at_prologue:
            thread.StepOver()
        self.hit_correct_line(
            thread, "This is where we will stop in the protocol dispatch")

        # Step out of the protocol function, one step out should also
        # get us past any dispatch thunk.
        thread.StepOut()
        stop_on_caller = self.hit_correct_line(thread, "indirect.protocol_func(20)", False)
        stop_at_cd_maker = self.hit_correct_line(thread, "var cd_maker", False)

        # In swift-4.0 U before, one step over is necessary because step out doesn't
        # finish off the line.
        # In swift-4.1 we now step over the line but we also stop on the "var cd_maker"
        # line, which we didn't with swift-4.0.  So we check for either of these.

        if stop_on_caller or stop_at_cd_maker:
            thread.StepOver()

        # Step over the assignment.
        stop_on_partial_apply = self.hit_correct_line(thread, "var cd_maker =", False)
        if stop_on_partial_apply:
            thread.StepOver()

        self.hit_correct_line(thread, "doSomethingWithFunction(cd_maker, 10)")

        thread.StepInto()
        stop_in_prologue = self.hit_correct_line(
            thread, "Stopped in doSomethingWithFunctionResult decl.", False)
        if stop_in_prologue:
            thread.StepOver()
        self.hit_correct_line(
            thread, "Calling doSomethingWithFunction with value")
        thread.StepOver()
        self.hit_correct_line(thread, "let result = f(other_value)")

        # Now try stepping into calling a closure, there's several
        # layers of goo to get through:
        thread.StepInto()
        stop_in_prologue = self.hit_correct_line(
            thread, "Step into cd_maker stops at closure decl instead.", False)
        if stop_in_prologue:
            thread.StepOver()

        self.hit_correct_line(thread, "Step into should stop here in closure.")

        # Then make sure we can get back out:
        thread.StepOut()
        # Again, step out may not have completed the source line we
        # stepped in FROM...
        stop_in_caller_line = self.hit_correct_line(
            thread, "let result = f(other_value)", False)
        if stop_in_caller_line:
            thread.StepOver()

        self.hit_correct_line(
            thread, "result.protocol_func(other_value)")

    def tearDown(self):
        self.dbg.HandleCommand(
            "settings clear target.process.thread.step-avoid-libraries")
        super(TestSwiftStepping, self).tearDown()
