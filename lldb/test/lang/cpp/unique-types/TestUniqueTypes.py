"""
Test that template instaniations of std::vector<long> and <short> in the same module have the correct types.
"""

import unittest2
import lldb
import lldbutil
from lldbtest import *

class UniqueTypesTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "unique-types")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        self.buildDsym()
        self.unique_types()

    @dwarf_test
    def test_with_dwarf(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""
        self.buildDwarf()
        self.unique_types()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number inside main.cpp.
        self.line = line_number("main.cpp",
          "// Set breakpoint here to verify that std::vector 'longs' and 'shorts' have unique types.")

    def unique_types(self):
        """Test for unique types of std::vector<long> and std::vector<short>."""

        compiler = self.getCompiler()
        compiler_basename = os.path.basename(compiler)
        if "clang" in compiler_basename and int(self.getCompilerVersion().split('.')[0]) < 3:
            self.skipTest("rdar://problem/9173060 lldb hangs while running unique-types for clang version < 3")

        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # GCC 4.6.3 (but not 4.4, 4.6.5 or 4.7) encodes two locations for the 'return 0' statement in main.cpp
        locs = 2 if "gcc" in compiler_basename and "4.6.3" in self.getCompilerVersion() else 1
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=locs, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # Do a "frame variable --show-types longs" and verify "long" is in each line of output.
        self.runCmd("frame variable --show-types longs")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line, closing brace, and messages about more variables than can be displayed.
            if not x or x == '}' or x == '...' or "Some of your variables have more members than the debugger will show by default" in x:
                continue
            self.expect(x, "Expect type 'long'", exe=False,
                substrs = ['long'])

        # Do a "frame variable --show-types shorts" and verify "short" is in each line of output.
        self.runCmd("frame variable --show-types shorts")
        output = self.res.GetOutput()
        for x in [line.strip() for line in output.split(os.linesep)]:
            # Skip empty line, closing brace, and messages about more variables than can be displayed.
            if not x or x == '}' or x == '...' or "Some of your variables have more members than the debugger will show by default" in x:
                continue
            self.expect(x, "Expect type 'short'", exe=False,
                substrs = ['short'])
        

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
