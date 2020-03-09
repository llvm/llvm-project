"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestDataFormatterLibcxxBitset(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)

        primes = [1]*300
        primes[0] = primes[1] = 0
        for i in range(2, len(primes)):
            for j in range(2*i, len(primes), i):
                primes[j] = 0
        self.primes = primes

    def check(self, name, size):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetNumChildren(), size)
        for i in range(size):
            child = var.GetChildAtIndex(i)
            self.assertEqual(child.GetValueAsUnsigned(), self.primes[i],
                    "variable: %s, index: %d"%(name, size))

    @add_test_categories(["libc++"])
    def test_value(self):
        """Test that std::bitset is displayed correctly"""
        self.build()
        lldbutil.run_to_source_breakpoint(self, '// break here',
                lldb.SBFileSpec("main.cpp", False))

        self.check("empty", 0)
        self.check("small", 13)
        self.check("large", 200)

    @add_test_categories(["libc++"])
    def test_ptr_and_ref(self):
        """Test that ref and ptr to std::bitset is displayed correctly"""
        self.build()
        (_, process, _, bkpt) = lldbutil.run_to_source_breakpoint(self,
                'Check ref and ptr',
                lldb.SBFileSpec("main.cpp", False))

        self.check("ref", 13)
        self.check("ptr", 13)

        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check("ref", 200)
        self.check("ptr", 200)
