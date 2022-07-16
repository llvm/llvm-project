# TestSwiftTypes.py
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
Test that we can inspect basic Swift types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import platform


class TestSwiftTypes(TestBase):

    @swiftTest
    def test_swift_types(self):
        """Test that we can inspect basic Swift types"""
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
            'Set breakpoint here', self.main_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame #0 should be at our breakpoint.
        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertTrue(len(threads) == 1)
        self.thread = threads[0]

        value_str = "value = -2"
        self.expect(
            "frame variable --raw int16_minus_two",
            substrs=[
                'Int16',
                'int16_minus_two',
                value_str])
        self.expect(
            "frame variable --raw int32_minus_two",
            substrs=[
                'Int32',
                'int32_minus_two',
                value_str])
        # TODO: change 'Int' back to 'Int64' name back when
        # <rdar://problem/15078795> is fixed
        self.expect(
            "frame variable --raw int64_minus_two",
            substrs=[
                'Int',
                'int64_minus_two',
                value_str])
        self.expect(
            "frame variable --raw int_minus_two",
            substrs=[
                'Int',
                'int_minus_two',
                value_str])
        value_str = "value = 2"
        self.expect(
            "frame variable --raw int8_plus_two",
            substrs=[
                'Int8',
                'int8_plus_two',
                value_str])
        self.expect(
            "frame variable --raw int16_plus_two",
            substrs=[
                'Int16',
                'int16_plus_two',
                value_str])
        self.expect(
            "frame variable --raw int32_plus_two",
            substrs=[
                'Int32',
                'int32_plus_two',
                value_str])
        # TODO: change 'Int' back to 'Int64' name back when
        # <rdar://problem/15078795> is fixed
        self.expect(
            "frame variable --raw int64_plus_two",
            substrs=[
                'Int',
                'int64_plus_two',
                value_str])
        self.expect(
            "frame variable --raw int_plus_two",
            substrs=[
                'Int',
                'int_plus_two',
                value_str])
        value_str = "value = 2"
        self.expect(
            "frame variable --raw int8_plus_two",
            substrs=[
                'Int8',
                'int8_plus_two',
                value_str])
        self.expect(
            "frame variable --raw int16_plus_two",
            substrs=[
                'Int16',
                'int16_plus_two',
                value_str])
        self.expect(
            "frame variable --raw int32_plus_two",
            substrs=[
                'Int32',
                'int32_plus_two',
                value_str])
        # TODO: change 'Int' back to 'Int64' name back when
        # <rdar://problem/15078795> is fixed
        self.expect(
            "frame variable --raw int64_plus_two",
            substrs=[
                'Int',
                'int64_plus_two',
                value_str])

        self.expect(
            "frame variable --raw float32",
            substrs=[
                'Float',
                'float32',
                'value = 1.25'])
        self.expect(
            "frame variable --raw float64",
            substrs=[
                'Double',
                'float64',
                'value = 2.5'])
        if self.getArchitecture() == "x86_64":
            self.expect(
                "frame variable --raw float80",
                substrs=[
                    'Float80',
                    'float80',
                    'value = 1.0625'])
        self.expect(
            "frame variable --raw float",
            substrs=[
                'Float',
                'float',
                'value = 3.75'])

        self.expect("frame variable --raw hello", substrs=['String'])

        self.expect("p/x int64_minus_two", substrs=['0xfffffffffffffffe'])
        self.expect("p/u ~1", substrs=['18446744073709551614'])
        self.expect("p/d ~1", substrs=['-2'])

        self.expect('frame variable uint8_max', substrs=['-1'], matching=False)
        self.expect(
            'frame variable uint16_max',
            substrs=['-1'],
            matching=False)
        self.expect(
            'frame variable uint32_max',
            substrs=['-1'],
            matching=False)
        self.expect(
            'frame variable uint64_max',
            substrs=['-1'],
            matching=False)
        self.expect('frame variable uint_max', substrs=['-1'], matching=False)

        self.expect('frame variable uint8_max', substrs=['255'])
        self.expect('frame variable uint16_max', substrs=['65535'])
        self.expect('frame variable uint32_max', substrs=['4294967295'])
        self.expect(
            'frame variable uint64_max',
            substrs=['18446744073709551615'])

