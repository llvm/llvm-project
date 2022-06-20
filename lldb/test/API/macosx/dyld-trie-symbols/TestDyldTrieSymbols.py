"""
Test that we read the exported symbols from the dyld trie
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class DyldTrieSymbolsTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @skipIfRemote
    @skipUnlessDarwin

    def test_dyld_trie_symbols(self):
        """Test that we make create symbol table entries from the dyld trie data structure."""
        self.build()
        unstripped_exe = self.getBuildArtifact("a.out")
        stripped_exe = self.getBuildArtifact("a.out-stripped")

        unstripped_target = self.dbg.CreateTarget(unstripped_exe)
        self.assertTrue(unstripped_target.IsValid(), "Got a vaid stripped target.")

        # Verify that the expected symbols are present in an unstripped
        # binary, and that we didn't duplicate the entries in the symbol
        # table.
        unstripped_bazval_symbols = unstripped_target.FindSymbols("bazval")
        self.assertEqual(unstripped_bazval_symbols.GetSize(), 1)
        unstripped_patval_symbols = unstripped_target.FindSymbols("patval")
        self.assertEqual(unstripped_patval_symbols.GetSize(), 1)
        unstripped_Z3foo_symbols = unstripped_target.FindSymbols("_Z3foov")
        self.assertEqual(unstripped_Z3foo_symbols.GetSize(), 1)
        unstripped_foo_symbols = unstripped_target.FindSymbols("foo")
        self.assertEqual(unstripped_foo_symbols.GetSize(), 1)

        # Make sure we can look up the mangled name and the demangled base
        # name.
        unstripped_Z3pat_symbols = unstripped_target.FindSymbols("_Z3pati")
        self.assertEqual(unstripped_Z3pat_symbols.GetSize(), 1)
        unstripped_pat_symbols = unstripped_target.FindSymbols("pat")
        self.assertEqual(unstripped_pat_symbols.GetSize(), 1)

        unstripped_bar_symbols = unstripped_target.FindSymbols("bar")
        self.assertEqual(unstripped_bar_symbols.GetSize(), 1)



        # Verify that we can retrieve all the symbols with external
        # linkage after the binary has been stripped; they should not
        # exist in the nlist records at this point and can only be
        # retrieved from the dyld trie structure.

        stripped_target = self.dbg.CreateTarget(stripped_exe)
        self.assertTrue(stripped_target.IsValid(), "Got a vaid stripped target.")

        # Check that we're able to still retrieve all the symbols after
        # the binary has been stripped. Check that one we know will be
        # removed is absent.
        stripped_bazval_symbols = stripped_target.FindSymbols("bazval")
        self.assertEqual(stripped_bazval_symbols.GetSize(), 1)
        stripped_patval_symbols = stripped_target.FindSymbols("patval")
        self.assertEqual(stripped_patval_symbols.GetSize(), 1)
        stripped_Z3foo_symbols = stripped_target.FindSymbols("_Z3foov")
        self.assertEqual(stripped_Z3foo_symbols.GetSize(), 1)
        stripped_foo_symbols = stripped_target.FindSymbols("foo")
        self.assertEqual(stripped_foo_symbols.GetSize(), 1)

        # make sure we can look up the mangled name, demangled base name,
        # demangled name with argument.
        stripped_Z3pat_symbols = stripped_target.FindSymbols("_Z3pati")
        self.assertEqual(stripped_Z3pat_symbols.GetSize(), 1)
        stripped_pat_symbols = stripped_target.FindSymbols("pat")
        self.assertEqual(stripped_pat_symbols.GetSize(), 1)

        # bar should have been strippped.  We should not find it, or the
        # stripping went wrong.
        stripped_bar_symbols = stripped_target.FindSymbols("bar")
        self.assertEqual(stripped_bar_symbols.GetSize(), 0)

        # confirm that we classified objc runtime symbols correctly and
        # stripped off the objc prefix from the symbol names.
        syms_ctx = stripped_target.FindSymbols("SourceBase")
        self.assertEqual(syms_ctx.GetSize(), 2)

        sym1 = syms_ctx.GetContextAtIndex(0).GetSymbol()
        sym2 = syms_ctx.GetContextAtIndex(1).GetSymbol()

        # one of these should be a lldb.eSymbolTypeObjCClass, the other
        # should be lldb.eSymbolTypeObjCMetaClass.
        if sym1.GetType() == lldb.eSymbolTypeObjCMetaClass:
            self.assertEqual(sym2.GetType(), lldb.eSymbolTypeObjCClass)
        else:
            if sym1.GetType() == lldb.eSymbolTypeObjCClass:
                self.assertEqual(sym2.GetType(), lldb.eSymbolTypeObjCMetaClass)
            else:
                self.assertTrue(sym1.GetType() == lldb.eSymbolTypeObjCMetaClass or sym1.GetType() == lldb.eSymbolTypeObjCClass)

        syms_ctx = stripped_target.FindSymbols("SourceDerived._derivedValue")
        self.assertEqual(syms_ctx.GetSize(), 1)
        sym = syms_ctx.GetContextAtIndex(0).GetSymbol()
        self.assertEqual(sym.GetType(), lldb.eSymbolTypeObjCIVar)
