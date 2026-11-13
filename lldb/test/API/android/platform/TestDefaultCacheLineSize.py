"""
Verify the default cache line size for android targets
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class DefaultCacheLineSizeTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessTargetAndroid
    def test_cache_line_size(self):
        self.build()
        _, process, _, _ = lldbutil.run_to_name_breakpoint(self, "main")

        # check the setting value
        self.expect(
            "settings show target.process.memory-cache-line-size", patterns=[" = 2048"]
        )

        # Run to completion.
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited, PROCESS_EXITED)
