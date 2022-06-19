# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSException(ObjCDataFormatterTestCase):

    def test_nsexception_with_run_command(self):
        """Test formatters for NSException."""
        self.appkit_tester_impl(self.nsexception_data_formatter_commands, True)

    @skipUnlessDarwin
    def test_nsexception_with_run_command_no_const(self):
        """Test formatters for NSException."""
        self.appkit_tester_impl(self.nsexception_data_formatter_commands, False)

    def nsexception_data_formatter_commands(self):
        self.expect(
            'frame variable except0 except1 except2 except3',
            substrs=[
                '(NSException *) except0 = ',
                '@"First"',
                '(NSException *) except1 = ',
                '@"Second"',
                '(NSException *) except2 = ',
                ' @"Third"',
                '(NSException *) except3 = ',
                ' @"Fourth"'
            ])
