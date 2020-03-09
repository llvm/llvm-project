# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSURL(ObjCDataFormatterTestCase):

    @skipUnlessDarwin
    def test_nsurl_with_run_command(self):
        """Test formatters for NSURL."""
        self.appkit_tester_impl(self.nsurl_data_formatter_commands)

    def nsurl_data_formatter_commands(self):
        self.expect(
            'frame variable cfurl_ref cfchildurl_ref cfgchildurl_ref',
            substrs=[
                '(CFURLRef) cfurl_ref = ', '@"http://www.foo.bar',
                'cfchildurl_ref = ', '@"page.html -- http://www.foo.bar',
                '(CFURLRef) cfgchildurl_ref = ',
                '@"?whatever -- http://www.foo.bar/page.html"'
            ])

        self.expect(
            'frame variable nsurl nsurl2 nsurl3',
            substrs=[
                '(NSURL *) nsurl = ', '@"http://www.foo.bar',
                '(NSURL *) nsurl2 =', '@"page.html -- http://www.foo.bar',
                '(NSURL *) nsurl3 = ',
                '@"?whatever -- http://www.foo.bar/page.html"'
            ])
