"""Test that SBValue clears non-addressable bits"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestClearSBValueNonAddressableBits(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # On AArch64 systems, the top bits that are not used for
    # addressing may be used for TBI, MTE, and/or pointer
    # authentication.
    @skipIf(archs=no_match(["aarch64", "arm64", "arm64e"]))

    # Only run this test on systems where TBI is known to be
    # enabled, so the address mask will clear the TBI bits.
    @skipUnlessPlatform(["linux"] + lldbplatformutil.getDarwinOSTriples())
    def test(self):
        self.source = "main.c"
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source, False)
        )

        if self.TraceOn():
            self.runCmd("frame variable")
            self.runCmd("frame variable &count &global")

        frame = thread.GetFrameAtIndex(0)

        count_p = frame.FindVariable("count_p")
        count_invalid_p = frame.FindVariable("count_invalid_p")
        self.assertEqual(
            count_p.GetValueAsUnsigned(), count_invalid_p.GetValueAsAddress()
        )
        self.assertNotEqual(
            count_invalid_p.GetValueAsUnsigned(), count_invalid_p.GetValueAsAddress()
        )
        self.assertEqual(5, count_p.Dereference().GetValueAsUnsigned())
        self.assertEqual(5, count_invalid_p.Dereference().GetValueAsUnsigned())

        global_p = frame.FindVariable("global_p")
        global_invalid_p = frame.FindVariable("global_invalid_p")
        self.assertEqual(
            global_p.GetValueAsUnsigned(), global_invalid_p.GetValueAsAddress()
        )
        self.assertNotEqual(
            global_invalid_p.GetValueAsUnsigned(), global_invalid_p.GetValueAsAddress()
        )
        self.assertEqual(10, global_p.Dereference().GetValueAsUnsigned())
        self.assertEqual(10, global_invalid_p.Dereference().GetValueAsUnsigned())

        main_p = frame.FindVariable("main_p")
        main_invalid_p = frame.FindVariable("main_invalid_p")
        self.assertEqual(
            main_p.GetValueAsUnsigned(), main_invalid_p.GetValueAsAddress()
        )
