"""
Test symbol table access for main.m.
"""

import os, time
import unittest2
import lldb
from lldbtest import *

@unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
class FoundationSymtabTestCase(TestBase):

    mydir = os.path.join("lang", "objc", "foundation")

    symbols_list = ['-[MyString initWithNSString:]',
                    '-[MyString dealloc]',
                    '-[MyString description]',
                    '-[MyString descriptionPauses]',     # synthesized property
                    '-[MyString setDescriptionPauses:]', # synthesized property
                    'Test_Selector',
                    'Test_NSString',
                    'Test_MyString',
                    'Test_NSArray',
                    'main'
                    ]

    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Test symbol table access with Python APIs."""
        self.buildDsym()
        self.symtab_access_python()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Test symbol table access with Python APIs."""
        self.buildDwarf()
        self.symtab_access_python()

    def symtab_access_python(self):
        """Test symbol table access with Python APIs."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        #
        # Exercise Python APIs to access the symbol table entries.
        #

        # Create the filespec by which to locate our a.out module.
        filespec = lldb.SBFileSpec(exe, False)

        module = target.FindModule(filespec)
        self.assertTrue(module, VALID_MODULE)

        # Create the set of known symbols.  As we iterate through the symbol
        # table, remove the symbol from the set if it is a known symbol.
        expected_symbols = set(self.symbols_list)
        for symbol in module:
            self.assertTrue(symbol, VALID_SYMBOL)
            #print "symbol:", symbol
            name = symbol.GetName()
            if name in expected_symbols:
                #print "Removing %s from known_symbols %s" % (name, expected_symbols)
                expected_symbols.remove(name)

        # At this point, the known_symbols set should have become an empty set.
        # If not, raise an error.
        #print "symbols unaccounted for:", expected_symbols
        self.assertTrue(len(expected_symbols) == 0,
                        "All the known symbols are accounted for")


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
