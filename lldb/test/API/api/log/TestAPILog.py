"""
Test API logging.
"""

import re

import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class APILogTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def test_api_log(self):
        """Test API logging"""
        logfile = os.path.join(self.getBuildDir(), "api-log.txt")

        def cleanup():
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)
        self.expect("log enable lldb api -f {}".format(logfile))

        self.dbg.SetDefaultArchitecture(None)
        self.dbg.GetScriptingLanguage(None)
        target = self.dbg.CreateTarget(None)

        print(logfile)
        with open(logfile, 'r') as f:
            log = f.read()

        # Find the SBDebugger's address.
        debugger_addr = re.findall(
            r"lldb::SBDebugger::GetScriptingLanguage\([^)]*\) \(0x([0-9a-fA-F]+),",
            log)

        # Make sure we've found a match.
        self.assertTrue(debugger_addr, log)

        # Make sure the GetScriptingLanguage matches.
        self.assertTrue(re.search(r'lldb::SBDebugger::GetScriptingLanguage\([^)]*\) \(0x{}, ""\)'.format(
            debugger_addr[0]), log), log)

        # Make sure the address matches.
        self.assertTrue(re.search(r'lldb::SBDebugger::CreateTarget\([^)]*\) \(0x{}, ""\)'.format(
            debugger_addr[0]), log), log)
