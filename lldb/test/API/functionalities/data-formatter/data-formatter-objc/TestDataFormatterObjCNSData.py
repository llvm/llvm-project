# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSData(ObjCDataFormatterTestCase):

    def test_nsdata_with_run_command(self):
        """Test formatters for  NSData."""
        self.appkit_tester_impl(self.nsdata_data_formatter_commands, True)

    @skipUnlessDarwin
    def test_nsdata_with_run_command_no_const(self):
        """Test formatters for  NSData."""
        self.appkit_tester_impl(self.nsdata_data_formatter_commands, False)

    def nsdata_data_formatter_commands(self):
        self.runCmd('settings set target.prefer-dynamic-value no-dynamic-values')

        self.expect(
            'frame variable immutableData mutableData data_ref mutable_data_ref mutable_string_ref concreteData concreteMutableData',
            substrs=[
                '(NSData *) immutableData = ', ' 5 bytes',
                '(NSData *) mutableData = ', ' 14 bytes',
                '(CFDataRef) data_ref = ', '@"5 bytes"',
                '(CFMutableDataRef) mutable_data_ref = ', '@"5 bytes"',
                '(CFMutableStringRef) mutable_string_ref = ',
                ' @"Wish ya knew"', '(NSData *) concreteData = ',
                ' 100000 bytes', '(NSMutableData *) concreteMutableData = ',
                ' 100000 bytes'
            ])
