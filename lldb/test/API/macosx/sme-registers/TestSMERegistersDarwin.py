import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSMERegistersDarwin(TestBase):
    NO_DEBUG_INFO_TESTCASE = True
    mydir = TestBase.compute_mydir(__file__)

    @skipIfRemote
    @skipUnlessDarwin
    @skipUnlessFeature("hw.optional.arm.FEAT_SME")
    @skipUnlessFeature("hw.optional.arm.FEAT_SME2")
    # thread_set_state/thread_get_state only avail in macOS 15.4+
    @skipIf(macos_version=["<", "15.4"])
    def test(self):
        """Test that we can read the contents of the SME/SVE registers on Darwin"""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break before sme", lldb.SBFileSpec("main.c")
        )
        frame = thread.GetFrameAtIndex(0)
        self.assertTrue(frame.IsValid())

        self.assertTrue(
            target.BreakpointCreateBySourceRegex(
                "break while sme", lldb.SBFileSpec("main.c")
            ).IsValid()
        )
        self.assertTrue(
            target.BreakpointCreateBySourceRegex(
                "break after sme", lldb.SBFileSpec("main.c")
            ).IsValid()
        )

        if self.TraceOn():
            self.runCmd("reg read -a")

        self.assertTrue(frame.register["svl"].GetError().Fail())
        self.assertTrue(frame.register["z0"].GetError().Fail())
        self.assertTrue(frame.register["p0"].GetError().Fail())
        self.assertTrue(frame.register["za"].GetError().Fail())
        self.assertTrue(frame.register["zt0"].GetError().Fail())

        process.Continue()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        # Now in SME enabled mode
        self.assertTrue(frame.register["svl"].GetError().Success())
        self.assertTrue(frame.register["z0"].GetError().Success())
        self.assertTrue(frame.register["p0"].GetError().Success())
        self.assertTrue(frame.register["za"].GetError().Success())
        self.assertTrue(frame.register["zt0"].GetError().Success())

        # SSVE and SME modes should be enabled (reflecting PSTATE.SM and PSTATE.ZA)
        svcr = frame.register["svcr"]
        self.assertEqual(svcr.GetValueAsUnsigned(), 3)

        svl_reg = frame.register["svl"]
        svl = svl_reg.GetValueAsUnsigned()

        z0 = frame.register["z0"]
        self.assertEqual(z0.GetNumChildren(), svl)
        self.assertEqual(z0.GetChildAtIndex(0).GetValueAsUnsigned(), 0x1)
        self.assertEqual(z0.GetChildAtIndex(svl - 1).GetValueAsUnsigned(), 0x1)

        z31 = frame.register["z31"]
        self.assertEqual(z31.GetNumChildren(), svl)
        self.assertEqual(z31.GetChildAtIndex(0).GetValueAsUnsigned(), 32)
        self.assertEqual(z31.GetChildAtIndex(svl - 1).GetValueAsUnsigned(), 32)

        p0 = frame.register["p0"]
        self.assertEqual(p0.GetNumChildren(), svl / 8)
        self.assertEqual(p0.GetChildAtIndex(0).GetValueAsUnsigned(), 0xFF)
        self.assertEqual(
            p0.GetChildAtIndex(p0.GetNumChildren() - 1).GetValueAsUnsigned(), 0xFF
        )

        p15 = frame.register["p15"]
        self.assertEqual(p15.GetNumChildren(), svl / 8)
        self.assertEqual(p15.GetChildAtIndex(0).GetValueAsUnsigned(), 0xFF)
        self.assertEqual(
            p15.GetChildAtIndex(p15.GetNumChildren() - 1).GetValueAsUnsigned(), 0xFF
        )

        za = frame.register["za"]
        self.assertEqual(za.GetNumChildren(), (svl * svl))
        za_0 = za.GetChildAtIndex(0)
        self.assertEqual(za_0.GetValueAsUnsigned(), 4)
        za_final = za.GetChildAtIndex(za.GetNumChildren() - 1)
        self.assertEqual(za_final.GetValueAsUnsigned(), 67)

        zt0 = frame.register["zt0"]
        self.assertEqual(zt0.GetNumChildren(), 64)
        zt0_0 = zt0.GetChildAtIndex(0)
        self.assertEqual(zt0_0.GetValueAsUnsigned(), 0)
        zt0_final = zt0.GetChildAtIndex(63)
        self.assertEqual(zt0_final.GetValueAsUnsigned(), 63)

        # Modify all of the registers, instruction step, confirm that the
        # registers have the new values.  Without the instruction step, it's
        # possible debugserver or lldb could lie about the write succeeding.

        z0_old_values = []
        z0_new_values = []
        z0_new_str = '"{'
        for i in range(svl):
            z0_old_values.append(z0.GetChildAtIndex(i).GetValueAsUnsigned())
            z0_new_values.append(z0_old_values[i] + 5)
            z0_new_str = z0_new_str + ("0x%02x " % z0_new_values[i])
        z0_new_str = z0_new_str + '}"'
        self.runCmd("reg write z0 %s" % z0_new_str)

        z31_old_values = []
        z31_new_values = []
        z31_new_str = '"{'
        for i in range(svl):
            z31_old_values.append(z31.GetChildAtIndex(i).GetValueAsUnsigned())
            z31_new_values.append(z31_old_values[i] + 3)
            z31_new_str = z31_new_str + ("0x%02x " % z31_new_values[i])
        z31_new_str = z31_new_str + '}"'
        self.runCmd("reg write z31 %s" % z31_new_str)

        p0_old_values = []
        p0_new_values = []
        p0_new_str = '"{'
        for i in range(int(svl / 8)):
            p0_old_values.append(p0.GetChildAtIndex(i).GetValueAsUnsigned())
            p0_new_values.append(p0_old_values[i] - 5)
            p0_new_str = p0_new_str + ("0x%02x " % p0_new_values[i])
        p0_new_str = p0_new_str + '}"'
        self.runCmd("reg write p0 %s" % p0_new_str)

        p15_old_values = []
        p15_new_values = []
        p15_new_str = '"{'
        for i in range(int(svl / 8)):
            p15_old_values.append(p15.GetChildAtIndex(i).GetValueAsUnsigned())
            p15_new_values.append(p15_old_values[i] - 8)
            p15_new_str = p15_new_str + ("0x%02x " % p15_new_values[i])
        p15_new_str = p15_new_str + '}"'
        self.runCmd("reg write p15 %s" % p15_new_str)

        za_old_values = []
        za_new_values = []
        za_new_str = '"{'
        for i in range(svl * svl):
            za_old_values.append(za.GetChildAtIndex(i).GetValueAsUnsigned())
            za_new_values.append(za_old_values[i] + 7)
            za_new_str = za_new_str + ("0x%02x " % za_new_values[i])
        za_new_str = za_new_str + '}"'
        self.runCmd("reg write za %s" % za_new_str)

        zt0_old_values = []
        zt0_new_values = []
        zt0_new_str = '"{'
        for i in range(64):
            zt0_old_values.append(zt0.GetChildAtIndex(i).GetValueAsUnsigned())
            zt0_new_values.append(zt0_old_values[i] + 2)
            zt0_new_str = zt0_new_str + ("0x%02x " % zt0_new_values[i])
        zt0_new_str = zt0_new_str + '}"'
        self.runCmd("reg write zt0 %s" % zt0_new_str)

        thread.StepInstruction(False)
        frame = thread.GetFrameAtIndex(0)

        if self.TraceOn():
            self.runCmd("reg read -a")

        z0 = frame.register["z0"]
        for i in range(z0.GetNumChildren()):
            self.assertEqual(
                z0_new_values[i], z0.GetChildAtIndex(i).GetValueAsUnsigned()
            )

        z31 = frame.register["z31"]
        for i in range(z31.GetNumChildren()):
            self.assertEqual(
                z31_new_values[i], z31.GetChildAtIndex(i).GetValueAsUnsigned()
            )

        p0 = frame.register["p0"]
        for i in range(p0.GetNumChildren()):
            self.assertEqual(
                p0_new_values[i], p0.GetChildAtIndex(i).GetValueAsUnsigned()
            )

        p15 = frame.register["p15"]
        for i in range(p15.GetNumChildren()):
            self.assertEqual(
                p15_new_values[i], p15.GetChildAtIndex(i).GetValueAsUnsigned()
            )

        za = frame.register["za"]
        for i in range(za.GetNumChildren()):
            self.assertEqual(
                za_new_values[i], za.GetChildAtIndex(i).GetValueAsUnsigned()
            )

        zt0 = frame.register["zt0"]
        for i in range(zt0.GetNumChildren()):
            self.assertEqual(
                zt0_new_values[i], zt0.GetChildAtIndex(i).GetValueAsUnsigned()
            )

        process.Continue()
        frame = thread.GetFrameAtIndex(0)
        self.assertEqual(thread.GetStopReason(), lldb.eStopReasonBreakpoint)

        self.assertTrue(frame.register["svl"].GetError().Fail())
        self.assertTrue(frame.register["z0"].GetError().Fail())
        self.assertTrue(frame.register["p0"].GetError().Fail())
        self.assertTrue(frame.register["za"].GetError().Fail())
        self.assertTrue(frame.register["zt0"].GetError().Fail())
