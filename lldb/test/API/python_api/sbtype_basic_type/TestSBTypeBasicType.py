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
