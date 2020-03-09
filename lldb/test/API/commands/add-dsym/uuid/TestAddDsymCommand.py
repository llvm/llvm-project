"""Test that the 'add-dsym', aka 'target symbols add', command informs the user about success or failure."""



import os
import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class AddDsymCommandCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.template = 'main.cpp.template'
        self.source = 'main.cpp'
        self.teardown_hook_added = False

    @no_debug_info_test
    def test_add_dsym_command_with_error(self):
        """Test that the 'add-dsym' command informs the user about failures."""

        # Call the program generator to produce main.cpp, version 1.
        self.generate_main_cpp(version=1)
        self.buildDefault(dictionary={'MAKE_DSYM':'YES'})

        # Insert some delay and then call the program generator to produce
        # main.cpp, version 2.
        time.sleep(5)
        self.generate_main_cpp(version=101)
        # Now call make again, but this time don't generate the dSYM.
        self.buildDefault(dictionary={'MAKE_DSYM':'NO'})

        self.exe_name = 'a.out'
        self.do_add_dsym_with_error(self.exe_name)

    @no_debug_info_test
    def test_add_dsym_command_with_success(self):
        """Test that the 'add-dsym' command informs the user about success."""

        # Call the program generator to produce main.cpp, version 1.
        self.generate_main_cpp(version=1)
        self.buildDefault(dictionary={'MAKE_DSYM':'YES'})

        self.exe_name = 'a.out'
        self.do_add_dsym_with_success(self.exe_name)

    @no_debug_info_test
    def test_add_dsym_with_dSYM_bundle(self):
        """Test that the 'add-dsym' command informs the user about success."""

        # Call the program generator to produce main.cpp, version 1.
        self.generate_main_cpp(version=1)
        self.buildDefault(dictionary={'MAKE_DSYM':'YES'})

        self.exe_name = 'a.out'
        self.do_add_dsym_with_dSYM_bundle(self.exe_name)

    def generate_main_cpp(self, version=0):
        """Generate main.cpp from main.cpp.template."""
        temp = os.path.join(self.getSourceDir(), self.template)
        with open(temp, 'r') as f:
            content = f.read()

        new_content = content.replace(
            '%ADD_EXTRA_CODE%',
            'printf("This is version %d\\n");' %
            version)
        src = os.path.join(self.getBuildDir(), self.source)
        with open(src, 'w') as f:
            f.write(new_content)

        # The main.cpp has been generated, add a teardown hook to remove it.
        if not self.teardown_hook_added:
            self.addTearDownHook(lambda: os.remove(src))
            self.teardown_hook_added = True

    def do_add_dsym_with_error(self, exe_name):
        """Test that the 'add-dsym' command informs the user about failures."""
        exe_path = self.getBuildArtifact(exe_name)
        self.runCmd("file " + exe_path, CURRENT_EXECUTABLE_SET)

        wrong_path = os.path.join(self.getBuildDir(),
                                  "%s.dSYM" % exe_name, "Contents")
        self.expect("add-dsym " + wrong_path, error=True,
                    substrs=['invalid module path'])

        right_path = os.path.join(
            self.getBuildDir(),
            "%s.dSYM" %
            exe_path,
            "Contents",
            "Resources",
            "DWARF",
            exe_name)
        self.expect("add-dsym " + right_path, error=True,
                    substrs=['symbol file', 'does not match'])

    def do_add_dsym_with_success(self, exe_name):
        """Test that the 'add-dsym' command informs the user about success."""
        exe_path = self.getBuildArtifact(exe_name)
        self.runCmd("file " + exe_path, CURRENT_EXECUTABLE_SET)

        # This time, the UUID should match and we expect some feedback from
        # lldb.
        right_path = os.path.join(
            self.getBuildDir(),
            "%s.dSYM" %
            exe_path,
            "Contents",
            "Resources",
            "DWARF",
            exe_name)
        self.expect("add-dsym " + right_path,
                    substrs=['symbol file', 'has been added to'])

    def do_add_dsym_with_dSYM_bundle(self, exe_name):
        """Test that the 'add-dsym' command informs the user about success when loading files in bundles."""
        exe_path = self.getBuildArtifact(exe_name)
        self.runCmd("file " + exe_path, CURRENT_EXECUTABLE_SET)

        # This time, the UUID should be found inside the bundle
        right_path = "%s.dSYM" % exe_path
        self.expect("add-dsym " + right_path,
                    substrs=['symbol file', 'has been added to'])
