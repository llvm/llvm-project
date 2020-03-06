"""
Test that template instaniations of std::vector<long> and <short> in the same module have the correct types.
"""



import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.lldbtest import *


class UniqueTypesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number inside main.cpp.
        self.line = line_number(
            "main.cpp",
            "// Set breakpoint here to verify that std::vector 'longs' and 'shorts' have unique types.")

    def test(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        self.build()

        compiler = self.getCompiler()
        compiler_basename = os.path.basename(compiler)

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=-1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # Do a "frame variable --show-types longs" and verify "long" is in each
        # line of output.
        self.runCmd("frame variable --show-types longs")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line, closing brace, and messages about more variables
            # than can be displayed.
            if not x or x == '}' or x == '...' or "Some of your variables have more members than the debugger will show by default" in x:
                continue
            self.expect(x, "Expect type 'long'", exe=False,
                        substrs=['long'])

        # Do a "frame variable --show-types shorts" and verify "short" is in
        # each line of output.
        self.runCmd("frame variable --show-types shorts")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line, closing brace, and messages about more variables
            # than can be displayed.
            if not x or x == '}' or x == '...' or "Some of your variables have more members than the debugger will show by default" in x:
                continue
            self.expect(x, "Expect type 'short'", exe=False,
                        substrs=['short'])
