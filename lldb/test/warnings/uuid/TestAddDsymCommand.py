"""Test that the 'add-dsym', aka 'target symbols add', command informs the user about success or failure."""

import os, time
import unittest2
import lldb
import pexpect
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class AddDsymCommandCase(TestBase):

    mydir = os.path.join("warnings", "uuid")

    def setUp(self):
        TestBase.setUp(self)
        self.template = 'main.cpp.template'
        self.source = 'main.cpp'
        self.teardown_hook_added = False

    def test_add_dsym_command_with_error(self):
        """Test that the 'add-dsym' command informs the user about failures."""

        # Call the program generator to produce main.cpp, version 1.
        self.generate_main_cpp(version=1)
        self.buildDsym(clean=True)

        # Insert some delay and then call the program generator to produce main.cpp, version 2.
        time.sleep(5)
        self.generate_main_cpp(version=101)
        # Now call make again, but this time don't generate the dSYM.
        self.buildDwarf(clean=False)

        self.exe_name = 'a.out'
        self.do_add_dsym_with_error(self.exe_name)

    def test_add_dsym_command_with_success(self):
        """Test that the 'add-dsym' command informs the user about success."""

        # Call the program generator to produce main.cpp, version 1.
        self.generate_main_cpp(version=1)
        self.buildDsym(clean=True)

        self.exe_name = 'a.out'
        self.do_add_dsym_with_success(self.exe_name)

    def test_add_dsym_with_dSYM_bundle(self):
        """Test that the 'add-dsym' command informs the user about success."""

        # Call the program generator to produce main.cpp, version 1.
        self.generate_main_cpp(version=1)
        self.buildDsym(clean=True)

        self.exe_name = 'a.out'
        self.do_add_dsym_with_dSYM_bundle(self.exe_name)


    def generate_main_cpp(self, version=0):
        """Generate main.cpp from main.cpp.template."""
        temp = os.path.join(os.getcwd(), self.template)
        with open(temp, 'r') as f:
            content = f.read()

        new_content = content.replace('%ADD_EXTRA_CODE%',
                                      'printf("This is version %d\\n");' % version)
        src = os.path.join(os.getcwd(), self.source)
        with open(src, 'w') as f:
            f.write(new_content)

        # The main.cpp has been generated, add a teardown hook to remove it.
        if not self.teardown_hook_added:
            self.addTearDownHook(lambda: os.remove(src))
            self.teardown_hook_added = True

    def do_add_dsym_with_error(self, exe_name):
        """Test that the 'add-dsym' command informs the user about failures."""
        self.runCmd("file " + exe_name, CURRENT_EXECUTABLE_SET)

        wrong_path = os.path.join("%s.dSYM" % exe_name, "Contents")
        self.expect("add-dsym " + wrong_path, error=True,
            substrs = ['invalid module path'])

        right_path = os.path.join("%s.dSYM" % exe_name, "Contents", "Resources", "DWARF", exe_name)
        self.expect("add-dsym " + right_path, error=True,
            substrs = ['symbol file', 'does not match'])

    def do_add_dsym_with_success(self, exe_name):
        """Test that the 'add-dsym' command informs the user about success."""
        self.runCmd("file " + exe_name, CURRENT_EXECUTABLE_SET)

        # This time, the UUID should match and we expect some feedback from lldb.
        right_path = os.path.join("%s.dSYM" % exe_name, "Contents", "Resources", "DWARF", exe_name)
        self.expect("add-dsym " + right_path,
            substrs = ['symbol file', 'has been added to'])

    def do_add_dsym_with_dSYM_bundle(self, exe_name):
        """Test that the 'add-dsym' command informs the user about success when loading files in bundles."""
        self.runCmd("file " + exe_name, CURRENT_EXECUTABLE_SET)

        # This time, the UUID should be found inside the bundle
        right_path = "%s.dSYM" % exe_name
        self.expect("add-dsym " + right_path,
            substrs = ['symbol file', 'has been added to'])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
