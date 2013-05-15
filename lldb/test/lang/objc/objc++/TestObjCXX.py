"""
Make sure that ivars of Objective-C++ classes are visible in LLDB.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class ObjCXXTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "objc++")

    @dsym_test
    def test_break_with_dsym(self):
        """Test ivars of Objective-C++ classes"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires Objective-C 2.0 runtime")
        self.buildDsym()
        self.do_testObjCXXClasses()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dwarf_test
    def test_break_with_dwarf(self):
        """Test ivars of Objective-C++ classes"""
        if self.getArchitecture() == 'i386':
            self.skipTest("requires Objective-C 2.0 runtime")
        self.buildDwarf()
        self.do_testObjCXXClasses()

    def do_testObjCXXClasses(self):
        """Test ivars of Objective-C++ classes"""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_source_regexp (self, 'breakpoint 1', num_expected_locations=1) 

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect("expr f->f", "Found ivar in class",
            substrs = ["= 3"])
        
if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
