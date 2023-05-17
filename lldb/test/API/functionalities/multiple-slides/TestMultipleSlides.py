"""
Test that a binary can be slid to different load addresses correctly
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MultipleSlidesTestCase(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    def test_mulitple_slides(self):
        """Test that a binary can be slid multiple times correctly."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        err = lldb.SBError()
        load_dependent_modules = False
        target = self.dbg.CreateTarget(exe, '', '', load_dependent_modules, err)
        self.assertTrue(target.IsValid())
        module = target.GetModuleAtIndex(0)
        self.assertTrue(module.IsValid())

        first_sym = target.FindSymbols("first").GetContextAtIndex(0).GetSymbol()
        second_sym = target.FindSymbols("second").GetContextAtIndex(0).GetSymbol()
        first_size = first_sym.GetEndAddress().GetOffset() - first_sym.GetStartAddress().GetOffset()
        second_size = second_sym.GetEndAddress().GetOffset() - second_sym.GetStartAddress().GetOffset()

        # View the first element of `first` and `second` while
        # they have no load address set.
        self.expect("expression/d ((int*)&first)[0]", substrs=['= 5'])
        self.expect("expression/d ((int*)&second)[0]", substrs=['= 6'])
        self.assertEqual(first_sym.GetStartAddress().GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)
        self.assertEqual(second_sym.GetStartAddress().GetLoadAddress(target), lldb.LLDB_INVALID_ADDRESS)

        # View the first element of `first` and `second` with
        # no slide applied, but with load address set.
        #
        # In memory, we have something like
        #    0x1000 - 0x17ff  first[]
        #    0x1800 - 0x1fff  second[]
        error = target.SetModuleLoadAddress(module, 0)
        self.assertSuccess(error)
        self.expect("expression/d ((int*)&first)[0]", substrs=['= 5'])
        self.expect("expression/d ((int*)&second)[0]", substrs=['= 6'])
        self.assertEqual(first_sym.GetStartAddress().GetLoadAddress(target), 
                         first_sym.GetStartAddress().GetFileAddress())
        self.assertEqual(second_sym.GetStartAddress().GetLoadAddress(target),
                         second_sym.GetStartAddress().GetFileAddress())

        # Slide it a little bit less than the size of the first array.
        #
        # In memory, we have something like
        #    0xfc0 - 0x17bf  first[]
        #    0x17c0 - 0x1fbf second[]
        #
        # but if the original entries are still present in lldb, 
        # the beginning address of second[] will get a load address
        # of 0x1800, instead of 0x17c0 (0x1800-64) as we need to get.
        error = target.SetModuleLoadAddress(module, first_size - 64)
        self.assertSuccess(error)
        self.expect("expression/d ((int*)&first)[0]", substrs=['= 5'])
        self.expect("expression/d ((int*)&second)[0]", substrs=['= 6'])
        self.assertNotEqual(first_sym.GetStartAddress().GetLoadAddress(target), 
                         first_sym.GetStartAddress().GetFileAddress())
        self.assertNotEqual(second_sym.GetStartAddress().GetLoadAddress(target),
                         second_sym.GetStartAddress().GetFileAddress())

        # Slide it back to the original vmaddr.
        error = target.SetModuleLoadAddress(module, 0)
        self.assertSuccess(error)
        self.expect("expression/d ((int*)&first)[0]", substrs=['= 5'])
        self.expect("expression/d ((int*)&second)[0]", substrs=['= 6'])
        self.assertEqual(first_sym.GetStartAddress().GetLoadAddress(target), 
                         first_sym.GetStartAddress().GetFileAddress())
        self.assertEqual(second_sym.GetStartAddress().GetLoadAddress(target),
                         second_sym.GetStartAddress().GetFileAddress())

        # Make sure we can use a slide > INT64_MAX.
        error = target.SetModuleLoadAddress(module, 0xffffffff12345678)
        self.assertSuccess(error)
