# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSError(ObjCDataFormatterTestCase):

    def test_nserror_with_run_command(self):
        """Test formatters for NSError."""
        self.appkit_tester_impl(self.nserror_data_formatter_commands)

    @expectedFailureAll(bugnumber='rdar://74106816')
    def nserror_data_formatter_commands(self):
        self.expect(
            'frame variable nserror', substrs=['domain: @"Foobar" - code: 12'])
         
        self.expect(
            'frame variable nserrorptr',
            substrs=['domain: @"Foobar" - code: 12'])

        # FIXME: <rdar://problem/25587546> On llvm.org this works without the `-d run`!
        self.expect(
            'frame variable -d run -- nserror->_userInfo', substrs=['2 key/value pairs'])

        self.expect(
            'frame variable nserror->_userInfo --ptr-depth 1 -d run-target',
            substrs=['@"a"', "1", '@"b"', "2"])
