"""
Test symbol table access for main.m.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class FoundationSymtabTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

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

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test symbol table access with Python APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Create the filespec by which to locate our a.out module.
        #
        #  - Use the absolute path to get the module for the current variant.
        #  - Use the relative path for reproducers. The modules are never
        #    orphaned because the SB objects are leaked intentionally. This
        #    causes LLDB to reuse the same module for every variant, because the
        #    UUID is the same for all the inferiors. FindModule below only
        #    compares paths and is oblivious to the fact that the UUIDs are the
        #    same.
        if configuration.is_reproducer():
            filespec = lldb.SBFileSpec('a.out', False)
        else:
            filespec = lldb.SBFileSpec(exe, False)

        module = target.FindModule(filespec)
        self.assertTrue(module, VALID_MODULE)

        # Create the set of known symbols.  As we iterate through the symbol
        # table, remove the symbol from the set if it is a known symbol.
        expected_symbols = set(self.symbols_list)
        for symbol in module:
            self.assertTrue(symbol, VALID_SYMBOL)
            self.trace("symbol:", symbol)
            name = symbol.GetName()
            if name in expected_symbols:
                self.trace("Removing %s from known_symbols %s" % (name, expected_symbols))
                expected_symbols.remove(name)

        # At this point, the known_symbols set should have become an empty set.
        # If not, raise an error.
        self.trace("symbols unaccounted for:", expected_symbols)
        self.assertTrue(len(expected_symbols) == 0,
                        "All the known symbols are accounted for")
