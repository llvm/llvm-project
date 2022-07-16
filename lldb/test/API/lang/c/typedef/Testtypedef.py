"""Look up type information for typedefs of same name at different lexical scope and check for correct display."""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypedefTestCase(TestBase):

    @expectedFailureAll(compiler="clang", bugnumber="llvm.org/pr19238")
    def test_typedef(self):
        """Test 'image lookup -t a' and check for correct display at different scopes."""
        self.build()
        self.image_lookup_for_multiple_typedefs()

    def image_lookup_for_multiple_typedefs(self):
        """Test 'image lookup -t a' at different scopes and check for correct display."""
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        typearray = (
            "float",
            "float",
            "char",
            "double *",
            "float",
            "int",
            "double",
            "float",
            "float")
        arraylen = len(typearray) + 1
        for i in range(1, arraylen):
            loc_line = line_number(
                'main.c', '// Set break point ' + str(i) + '.')
            lldbutil.run_break_set_by_file_and_line(
                self, "main.c", loc_line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        for t in typearray:
            self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                        substrs=['stopped', 'stop reason = breakpoint'])
            self.expect("image lookup -t a", DATA_TYPES_DISPLAYED_CORRECTLY,
                        substrs=['name = "' + t + '"'])
            self.runCmd("continue")
