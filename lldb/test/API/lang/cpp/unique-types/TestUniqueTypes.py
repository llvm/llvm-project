"""
Test that template instaniations of std::vector<long> and <short> in the same module have the correct types.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class UniqueTypesTestCase(TestBase):
    def test(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        self.build()
        lldbutil.run_to_source_breakpoint(
            self, "// Set breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        # Do a "frame variable --show-types longs" and verify "long" is in each
        # line of output.
        self.runCmd("frame variable --show-types longs")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line, closing brace, and messages about more variables
            # than can be displayed.
            if (
                not x
                or x == "}"
                or x == "..."
                or "Some of your variables have more members than the debugger will show by default"
                in x
            ):
                continue
            self.expect(x, "Expect type 'long'", exe=False, substrs=["long"])

        # Do a "frame variable --show-types shorts" and verify "short" is in
        # each line of output.
        self.runCmd("frame variable --show-types shorts")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line, closing brace, and messages about more variables
            # than can be displayed.
            if (
                not x
                or x == "}"
                or x == "..."
                or "Some of your variables have more members than the debugger will show by default"
                in x
            ):
                continue
            self.expect(x, "Expect type 'short'", exe=False, substrs=["short"])
