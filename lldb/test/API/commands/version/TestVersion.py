import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class VersionTestCase(TestBase):

    @no_debug_info_test
    def test_version(self):
        # Should work even when people patch the output,
        # so let's just assume that every vendor at least mentions
        # 'lldb' in their version string.
        self.expect("version", substrs=['lldb'])

    @no_debug_info_test
    def test_version_invalid_invocation(self):
        self.expect("version a", error=True,
                    substrs=["'version' doesn't take any arguments."])
