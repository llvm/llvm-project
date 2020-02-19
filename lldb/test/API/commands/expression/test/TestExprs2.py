"""
Test some more expression commands.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ExprCommands2TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.c.
        self.line = line_number(
            'main.cpp',
            '// Please test many expressions while stopped at this line:')

    def test_more_expr_commands(self):
        """Test some more expression commands."""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # Does static casting work?
        self.expect("expression (int*)argv",
                    startstr="(int *) $0 = 0x")
        # (int *) $0 = 0x00007fff5fbff258

        # Do return values containing the contents of expression locals work?
        self.expect("expression int i = 5; i",
                    startstr="(int) $1 = 5")
        # (int) $2 = 5
        self.expect("expression $1 + 1",
                    startstr="(int) $2 = 6")
        # (int) $3 = 6

        # Do return values containing the results of static expressions work?
        self.expect("expression 20 + 3",
                    startstr="(int) $3 = 23")
        # (int) $4 = 5
        self.expect("expression $3 + 1",
                    startstr="(int) $4 = 24")
        # (int) $5 = 6

    @skipIfLinux
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr24489")
    def test_expr_symbols(self):
        """Test symbols."""
        self.build()

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
                self, "main.cpp", self.line, num_expected_locations=1, loc_exact=False)

        self.runCmd("run", RUN_SUCCEEDED)

        # Do anonymous symbols work?
        self.expect("expression ((char**)environ)[0]",
                startstr="(char *) $0 = 0x")
        # (char *) $1 = 0x00007fff5fbff298 "Apple_PubSub_Socket_Render=/tmp/launch-7AEsUD/Render"
