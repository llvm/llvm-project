# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""

from __future__ import print_function

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSNumber(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_nsnumber_with_run_command(self):
        """Test formatters for  NS container classes."""
        self.appkit_tester_impl(self.nscontainers_data_formatter_commands, True)

    @skipUnlessDarwin
    def test_nsnumber_with_run_command_no_const(self):
        """Test formatters for  NS container classes."""
        self.appkit_tester_impl(self.nscontainers_data_formatter_commands, False)

    def nscontainers_data_formatter_commands(self):
        self.expect(
            'frame variable newArray nsDictionary newDictionary nscfDictionary cfDictionaryRef newMutableDictionary cfarray_ref mutable_array_ref',
            substrs=[
                '(NSArray *) newArray = ', '@"50 elements"',
                '(NSDictionary *) nsDictionary = ', ' 2 key/value pairs',
                '(NSDictionary *) newDictionary = ', ' 12 key/value pairs',
                '(CFDictionaryRef) cfDictionaryRef = ', ' 2 key/value pairs',
                '(NSDictionary *) newMutableDictionary = ', ' 21 key/value pairs',
                '(CFArrayRef) cfarray_ref = ', '@"3 elements"',
                '(CFMutableArrayRef) mutable_array_ref = ', '@"11 elements"'
            ])

        numbers = [ ("num1", "(int)5"),
                    ("num2", "(float)3.140000"),
                    ("num3", "(double)3.14"),
                    ("num4", "(int128_t)18446744073709551614"),
                    ("num5", "(char)65"),
                    ("num6", "(long)255"),
                    ("num7", "(long)2000000"),
                    ("num8_Y", "YES"),
                    ("num8_N", "NO"),
                    ("num9", "(short)-31616"),
                    ("num_at1", "(int)12"),
                    ("num_at2", "(int)-12"),
                    ("num_at3", "(double)12.5"),
                    ("num_at4", "(double)-12.5"),
                    ("num_at5", "(char)97"),
                    ("num_at6", "(float)42.123"),
                    ("num_at7", "(double)43.123"),
                    ("num_at8", "(long)12345"),
                    ("num_at9", "17375808098308635870"),
                    ("num_at9b", "-1070935975400915746"),
                    ("num_at10", "YES"),
                    ("num_at11", "NO"),
        ]

        for var, res in numbers:
            self.expect('frame variable ' + var, substrs=[res])

