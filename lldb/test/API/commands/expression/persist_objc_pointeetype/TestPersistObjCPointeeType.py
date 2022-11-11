"""
Test that we can p *objcObject
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class PersistObjCPointeeType(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.m', '// break here')

    @skipIf(archs=["i386", "i686"])
    @skipIf(debug_info="gmodules", archs=['arm64', 'armv7', 'armv7k', 'arm64e', 'arm64_32'])  # compile error with gmodules for iOS
    @add_test_categories(["objc"])
    def test_with(self):
        """Test that we can p *objcObject"""
        self.build()

        def cleanup():
            pass

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("p *self", substrs=['_sc_name = nil',
                                        '_sc_name2 = nil',
                                        '_sc_name3 = nil',
                                        '_sc_name4 = nil',
                                        '_sc_name5 = nil',
                                        '_sc_name6 = nil',
                                        '_sc_name7 = nil',
                                        '_sc_name8 = nil'])
