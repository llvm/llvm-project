# TestSwiftFixIts.py
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
Test getting return values
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftFixIts(TestBase):
    @skipEmbeddedSwift
    @swiftTest
    def test_swift_fixits(self):
        """Test applying fixits to expressions"""
        self.build()
        self.do_test()

    def setUp(self):
        TestBase.setUp(self)
        self.main_source = "main.swift"
        self.main_source_spec = lldb.SBFileSpec(self.main_source)

    def do_test(self):
        """Tests that we can break and display simple types"""
        exe_name = "a.out"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here to test fixits', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        self.process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(self.process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            self.process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        frame = self.thread.frames[0]

        options = lldb.SBExpressionOptions()
        options.SetAutoApplyFixIts(False)

        # First make sure the expressions fail with no fixits:
        value = frame.EvaluateExpression(
            "var $tmp : Int = does_have.could_be!!", options)
        self.assertTrue(value.GetError().Fail())
        self.assertTrue(
            value.GetError().GetError() != 0x1001,
            'Failure was not "no return type"')

        value = frame.EvaluateExpression(
                "var $tmp2 = wrapper.wrapped", options)
        self.assertTrue(value.GetError().Fail())
        self.assertTrue(
            value.GetError().GetError() != 0x1001,
            'Failure was not "no return type"')

        # Now turn on auto apply:
        options.SetAutoApplyFixIts(True)
        value = frame.EvaluateExpression(
            "var $tmp : Int = does_have.could_be!!", options)
        self.assertTrue(value.GetError().Fail(),
                        "There's no result so this is counted a fail.")
        self.assertTrue(
            value.GetError().GetError() == 0x1001,
            'This error is "no return type"')

        # Check that the expressions were correct:
        tmp_value = frame.EvaluateExpression("$tmp == 100")
        self.assertSuccess(tmp_value.GetError())
        self.assertTrue(tmp_value.GetSummary() == 'true')
        
        value = frame.EvaluateExpression(
                "var $tmp2 = wrapper.wrapped", options)
        tmp_value = frame.EvaluateExpression("$tmp2 == 7")
        self.assertSuccess(tmp_value.GetError())
        self.assertTrue(tmp_value.GetSummary() == 'true')

        obj_value = frame.var("obj")
        self.assertSuccess(obj_value.GetError())
        obj_addr = obj_value.GetValueAsUnsigned()
        self.assertNotEqual(obj_addr, 0)

        # Test LLDB-specific casting rewrites.
        for op in ("as", "as?", "as!"):
            ret = lldb.SBCommandReturnObject()
            self.ci.HandleCommand(f"expression 0x{obj_addr:x} {op} SomeClass", ret)
            self.assertTrue(ret.Succeeded())
            errors = ret.GetError()
            self.assertIn("Evaluated this expression after applying Fix-It(s):", errors)
            self.assertRegex(errors, r"unsafeBitCast\(.+, to: SomeClass.self\)")

        # Test LLDB-specific casting rewrite where the cast is a substring of a
        # larger expression, in this case a cast followed by property access.
        ret = lldb.SBCommandReturnObject()
        self.ci.HandleCommand(f"expression (0x{obj_addr:x} as SomeClass).value", ret)
        self.assertTrue(ret.Succeeded())
        self.assertRegex(
            ret.GetError(), r"\(unsafeBitCast\(.+, to: SomeClass.self\)\).value"
        )
        self.assertIn(" = 42", ret.GetOutput())
