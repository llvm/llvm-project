"""
Test lldb data formatter subsystem for bitset for libcxx and libstdcpp.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

VALUE = "VALUE"
REFERENCE = "REFERENCE"
POINTER = "POINTER"


class GenericBitsetDataFormatterTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def setUp(self):
        TestBase.setUp(self)
        primes = [1] * 1000
        primes[0] = primes[1] = 0
        for i in range(2, len(primes)):
            for j in range(2 * i, len(primes), i):
                primes[j] = 0
        self.primes = primes

    def getBitsetVariant(self, size, variant):
        if variant == VALUE:
            return "std::bitset<" + str(size) + ">"
        elif variant == REFERENCE:
            return "std::bitset<" + str(size) + "> &"
        elif variant == POINTER:
            return "std::bitset<" + str(size) + "> *"
        return ""

    def check(self, name, size, variant):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        self.assertEqual(var.GetNumChildren(), size)
        children = []
        for i in range(size):
            child = var.GetChildAtIndex(i)
            children.append(
                ValueCheck(value=str(bool(child.GetValueAsUnsigned())).lower())
            )
            self.assertEqual(
                child.GetValueAsUnsigned(),
                self.primes[i],
                "variable: %s, index: %d" % (name, i),
            )
        self.expect_var_path(
            name, type=self.getBitsetVariant(size, variant), children=children
        )

    def do_test_value(self):
        """Test that std::bitset is displayed correctly"""
        lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.cpp", False)
        )

        self.check("empty", 0, VALUE)
        self.check("small", 13, VALUE)
        self.check("medium", 70, VALUE)
        self.check("large", 1000, VALUE)

    @add_test_categories(["libstdcxx"])
    def test_value_libstdcpp(self):
        self.build(dictionary={'USE_LIBSTDCPP': "1"})
        self.do_test_value()

    @add_test_categories(["libc++"])
    def test_value_libcpp(self):
        self.build(dictionary={'USE_LIBCPP': "1"})
        self.do_test_value()

    @add_test_categories(["msvcstl"])
    def test_value_msvcstl(self):
        self.build()
        self.do_test_value()

    def do_test_ptr_and_ref(self):
        """Test that ref and ptr to std::bitset is displayed correctly"""
        (_, process, _, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Check ref and ptr", lldb.SBFileSpec("main.cpp", False)
        )

        self.check("ref", 13, REFERENCE)
        self.check("ptr", 13, POINTER)

        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check("ref", 70, REFERENCE)
        self.check("ptr", 70, POINTER)

        lldbutil.continue_to_breakpoint(process, bkpt)

        self.check("ref", 1000, REFERENCE)
        self.check("ptr", 1000, POINTER)

    @add_test_categories(["libstdcxx"])
    def test_ptr_and_ref_libstdcpp(self):
        self.build(dictionary={'USE_LIBSTDCPP': "1"})
        self.do_test_ptr_and_ref()

    @add_test_categories(["libc++"])
    def test_ptr_and_ref_libcpp(self):
        self.build(dictionary={'USE_LIBCPP': "1"})
        self.do_test_ptr_and_ref()

    @add_test_categories(["msvcstl"])
    def test_ptr_and_ref_msvcstl(self):
        self.build()
        self.do_test_ptr_and_ref()
