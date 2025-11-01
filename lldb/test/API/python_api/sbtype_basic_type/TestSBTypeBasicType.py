import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestCase(TestBase):
    def test(self):
        """Test that SBType.GetBasicType unwraps typedefs."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return", lldb.SBFileSpec("main.cpp"))

        a = self.frame().FindVariable("a")
        self.assertTrue(a)

        int_basic_type = a.GetType().GetBasicType()
        self.assertEqual(int_basic_type, 13)

        b = self.frame().FindVariable("b")
        self.assertTrue(b)

        c = self.frame().FindVariable("c")
        self.assertTrue(c)

        d = self.frame().FindVariable("d")
        self.assertTrue(d)

        self.assertEqual(b.GetType().GetBasicType(), int_basic_type)
        self.assertEqual(c.GetType().GetBasicType(), int_basic_type)
        self.assertEqual(d.GetType().GetBasicType(), int_basic_type)

        # Check the size of the chosen basic types.
        self.assertEqual(self.target().FindFirstType("__int128").size, 16)
        self.assertEqual(self.target().FindFirstType("unsigned __int128").size, 16)

        # Check the size of the chosen aliases of basic types.
        self.assertEqual(self.target().FindFirstType("__int128_t").size, 16)
        self.assertEqual(self.target().FindFirstType("__uint128_t").size, 16)

        # "_BitInt(...)" and "unsigned _BitInt(...)" are GNU C compiler extensions
        # that are supported by LLVM C(++) compiler as well.
        #
        # We check that LLDB is able to map the names of these types
        # (as reported by LLDB for variables of this type)
        # to the corresponding SBType objects.
        self.assertEqual(self.target().FindFirstType("_BitInt(65)").name, "_BitInt(65)")
        self.assertEqual(self.target().FindFirstType("_BitInt(65)").size, 16)
        self.assertEqual(
            self.target().FindFirstType("unsigned _BitInt(65)").name,
            "unsigned _BitInt(65)",
        )
        self.assertEqual(self.target().FindFirstType("unsigned _BitInt(65)").size, 16)
