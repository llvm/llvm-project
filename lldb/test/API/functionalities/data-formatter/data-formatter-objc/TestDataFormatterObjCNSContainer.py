# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSContainer(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_nscontainers_with_run_command(self):
        """Test formatters for  NS container classes."""
        self.appkit_tester_impl(self.nscontainers_data_formatter_commands)

    def nscontainers_data_formatter_commands(self):
        self.expect(
            'frame variable newArray nsDictionary newDictionary nscfDictionary cfDictionaryRef newMutableDictionary cfarray_ref mutable_array_ref',
            substrs=[
                '(NSArray *) newArray = ', '@"50 elements"',
                '(NSDictionary *) newDictionary = ', ' 12 key/value pairs',
                '(NSDictionary *) newMutableDictionary = ',
                ' 21 key/value pairs', '(NSDictionary *) nsDictionary = ',
                ' 2 key/value pairs', '(CFDictionaryRef) cfDictionaryRef = ',
                ' 3 key/value pairs', '(CFArrayRef) cfarray_ref = ',
                '@"3 elements"', '(CFMutableArrayRef) mutable_array_ref = ',
                '@"11 elements"'
            ])

        self.expect(
            'frame variable iset1 iset2 imset',
            substrs=['4 indexes', '512 indexes', '10 indexes'])

        self.expect(
            'frame variable binheap_ref',
            substrs=['(CFBinaryHeapRef) binheap_ref = ', '@"21 items"'])

        self.expect(
            'expression -d run -- (NSArray*)[NSArray new]',
            substrs=['@"0 elements"'])
