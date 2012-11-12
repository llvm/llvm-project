"""
Test that clang produces the __apple accelerator tables, for example, __apple_types, correctly.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
from lldbutil import symbol_type_to_str

class AppleTypesTestCase(TestBase):

    mydir = os.path.join("macosx", "debug-info", "apple_types")

    #rdar://problem/11166975
    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_debug_info_for_apple_types(self):
        """Test that __apple_types section does get produced by clang."""

        if not self.getCompiler().endswith('clang'):
            self.skipTest("clang compiler only test")

        self.buildDefault()
        self.apple_types(dot_o=True)

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_debug_info_for_apple_types_dsym(self):
        """Test that __apple_types section does get produced by dsymutil.
           This is supposed to succeed even with rdar://problem/11166975."""

        if not self.getCompiler().endswith('clang'):
            self.skipTest("clang compiler only test")

        self.buildDsym()
        self.apple_types(dot_o=False)

    def apple_types(self, dot_o):
        """Test that __apple_types section does get produced by clang."""
        if dot_o:
            exe = os.path.join(os.getcwd(), "main.o")
        else:
            exe = os.path.join(os.getcwd(), "a.out.dSYM/Contents/Resources/DWARF/a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)
        self.assertTrue(target.GetNumModules() > 0)

        # Hide stdout if not running with '-t' option.
        if not self.TraceOn():
            self.HideStdout()

        print "Number of modules for the target: %d" % target.GetNumModules()
        for module in target.module_iter():
            print module

        # Get the executable module at index 0.
        exe_module = target.GetModuleAtIndex(0)

        dwarf_section = exe_module.FindSection("__DWARF")
        self.assertTrue(dwarf_section)
        print "__DWARF section:", dwarf_section
        print "Number of sub-sections: %d" % dwarf_section.GetNumSubSections()
        INDENT = ' ' * 4
        for subsec in dwarf_section:
            print INDENT + str(subsec)

        debug_str_sub_section = dwarf_section.FindSubSection("__debug_str")
        self.assertTrue(debug_str_sub_section)
        print "__debug_str sub-section:", debug_str_sub_section

        # Find our __apple_types section by name.
        apple_types_sub_section = dwarf_section.FindSubSection("__apple_types")
        self.assertTrue(apple_types_sub_section)
        print "__apple_types sub-section:", apple_types_sub_section

        # These other three all important subsections should also be present.
        self.assertTrue(dwarf_section.FindSubSection("__apple_names") and
                        dwarf_section.FindSubSection("__apple_namespac") and
                        dwarf_section.FindSubSection("__apple_objc"))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
