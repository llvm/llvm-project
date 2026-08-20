# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterNSBundle(ObjCDataFormatterTestCase):
    SHARED_BUILD_TESTCASE = False

    # On macOS >= 27 Foundation caches the path of a URL-initialized NSBundle
    # as a native Swift String, which can only be formatted with the Swift
    # language plugin.
    @requiresSwiftPlugin(min_apple_os_version=27.0)
    def test_nsbundle_with_run_command(self):
        """Test formatters for NSBundle."""
        self.appkit_tester_impl(self.nsbundle_data_formatter_commands, True)

    @skipUnlessDarwin
    @requiresSwiftPlugin(min_apple_os_version=27.0)
    def test_nsbundle_with_run_command_no_sonct(self):
        """Test formatters for NSBundle."""
        self.appkit_tester_impl(self.nsbundle_data_formatter_commands, False)

    def nsbundle_data_formatter_commands(self):
        self.expect(
            "frame variable bundle_string bundle_url main_bundle",
            substrs=[
                "(NSBundle *) bundle_string = ",
                ' @"/System/Library/Frameworks/Accelerate.framework"',
                "(NSBundle *) bundle_url = ",
                '"/System/Library/Frameworks/Foundation.framework"',
                "(NSBundle *) main_bundle = ",
                "data-formatter-objc",
            ],
        )
