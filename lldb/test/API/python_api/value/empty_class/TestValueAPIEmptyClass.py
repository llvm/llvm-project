import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ValueAPIEmptyClassTestCase(TestBase):
    def test(self):
        self.build()
        line = line_number("main.cpp", "// Break at this line")

        _, _, thread, _ = lldbutil.run_to_line_breakpoint(
            self, lldb.SBFileSpec("main.cpp"), line
        )
        frame0 = thread.GetFrameAtIndex(0)

        # Verify that we can access to a frame variable with an empty class type
        e = frame0.FindVariable("e")
        self.assertTrue(e.IsValid(), VALID_VARIABLE)
        self.DebugSBValue(e)
        self.assertEqual(e.GetNumChildren(), 0)

        # Verify that we can acces to a frame variable what is a pointer to an
        # empty class
        ep = frame0.FindVariable("ep")
        self.assertTrue(ep.IsValid(), VALID_VARIABLE)
        self.DebugSBValue(ep)

        # Verify that we can dereference a pointer to an empty class
        epd = ep.Dereference()
        self.assertTrue(epd.IsValid(), VALID_VARIABLE)
        self.DebugSBValue(epd)
        self.assertEqual(epd.GetNumChildren(), 0)
