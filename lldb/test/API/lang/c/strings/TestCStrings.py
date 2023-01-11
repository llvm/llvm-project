"""
Tests that C strings work as expected in expressions
"""
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CStringsTestCase(TestBase):

    def test_with_run_command(self):
        """Tests that C strings work as expected in expressions"""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        line = line_number('main.c', '// breakpoint 1')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("process launch", RUN_SUCCEEDED)

        self.expect("expression -- a[2]",
                    patterns=["\((const )?char\) \$0 = 'c'"])

        self.expect("expression -- z[2]",
                    startstr="(const char) $1 = 'x'")

        # On Linux, the expression below will test GNU indirect function calls.
        self.expect("expression -- (int)strlen(\"hello\")",
                    startstr="(int) $2 = 5")

        self.expect("expression -- \"world\"[2]",
                    startstr="(const char) $3 = 'r'")

        self.expect("expression -- \"\"[0]",
                    startstr="(const char) $4 = '\\0'")

        self.expect("expr --raw -- \"hello\"",
                    substrs=['[0] = \'h\'',
                             '[5] = \'\\0\''])

        self.expect("expression \"hello\"",
                    substrs=['[6]) $', 'hello'])

        self.expect("expression (char*)\"hello\"",
                    substrs=['(char *) $', ' = 0x',
                             'hello'])

        self.expect("expression (int)strlen(\"\")",
                    substrs=['(int) $', ' = 0'])

        self.expect("expression !z",
                    substrs=['false'])
