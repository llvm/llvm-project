"""
Test lldb behaves sanely when formatting corrupted `std::string`s.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxInvalidStringDataFormatterTestCase(TestBase):
    @add_test_categories(["libc++"])
    @skipIf(oslist=[lldbplatformutil.getDarwinOSTriples()], archs=["arm", "aarch64"])
    def test(self):
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", lldb.SBFileSpec("main.cpp")
        )
        frame = thread.frames[0]

        if not self.process().GetAddressByteSize() == 8:
            self.skipTest("The test requires a 64-bit process")

        # The test assumes that std::string is in its cap-size-data layout.
        self.expect(
            "frame variable garbage1", substrs=["garbage1 = Summary Unavailable"]
        )
        self.expect(
            "frame variable garbage2", substrs=[r'garbage2 = "\xfa\xfa\xfa\xfa"']
        )
        self.expect("frame variable garbage3", substrs=[r'garbage3 = "\xf0\xf0"'])
        self.expect(
            "frame variable garbage4", substrs=["garbage4 = Summary Unavailable"]
        )
        self.expect(
            "frame variable garbage5", substrs=["garbage5 = Summary Unavailable"]
        )
