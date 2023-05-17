"""
Make sure we can find the binary inside an app bundle.
"""

import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import lldbsuite.test.lldbplatformutil as lldbplatformutil
from lldbsuite.test.lldbtest import *

@decorators.skipUnlessDarwin
class FindAppInMacOSAppBundle(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def test_find_app_in_bundle(self):
        """This reads in the .app, makes sure we get the right binary and can run it."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, 'breakpoint here', lldb.SBFileSpec('main.c'),
            exe_name=self.getBuildArtifact("TestApp.app"))
