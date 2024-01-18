# TestSwiftGetValueAsUnsigned.py
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
Tests that the SBValue::GetValueAsUnsigned() API works for Swift types
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class SwiftGetValueAsUnsignedAPITest(TestBase):
    @swiftTest
    def test_get_value_as_unsigned_sbapi(self):
        """Tests that the SBValue::GetValueAsUnsigned() API works for Swift types"""
        self.build()
        self.getvalue_commands()

    def getvalue_commands(self):
        """Tests that the SBValue::GetValueAsUnsigned() API works for Swift types"""
        (target, process, thread, breakpoint) = lldbutil.run_to_source_breakpoint(self, 
                "break here", lldb.SBFileSpec("main.swift"))

        frame = thread.GetFrameAtIndex(0)
        string = frame.FindVariable("aString")
        number = frame.FindVariable("aNumber")
        number.SetPreferSyntheticValue(True)
        classobject = frame.FindVariable("aClassObject")

        numberValue = number.GetValueAsUnsigned()
        self.assertTrue(
            numberValue == 123456,
            "Swift.Int does not have a valid value")

        builtinPointerValue = string.GetChildMemberWithName(
            "str_value").GetChildMemberWithName("base").GetChildMemberWithName("value")
        self.assertTrue(builtinPointerValue != 0,
                        "Builtin.RawPointer does not have a valid value")

        objectPointerValue = string.GetChildMemberWithName(
            "str_value").GetChildMemberWithName("value")
        self.assertTrue(objectPointerValue != 0,
                        "Builtin.RawPointer does not have a valid value")

        classValue = classobject.GetValueAsUnsigned()
        self.assertTrue(
            classValue != 0,
            "Class types are aggregates with pointer values")

