"""Test that lldb on Darwin ignores metadata in the top byte of addresses, both corefile and live."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestTBIHonored(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def do_variable_access_tests(self, frame):
        self.assertEqual(
            frame.variables["pb"][0]
            .GetChildMemberWithName("p")
            .Dereference()
            .GetValueAsUnsigned(),
            15,
        )
        addr = frame.variables["pb"][0].GetChildMemberWithName("p").GetValueAsUnsigned()
        # Confirm that there is metadata in the top byte of our pointer
        self.assertEqual((addr >> 56) & 0xFF, 0xFE)
        self.expect("expr -- *pb.p", substrs=["15"])
        self.expect("frame variable *pb.p", substrs=["15"])
        self.expect("expr -- *(int*)0x%x" % addr, substrs=["15"])

    # This test is valid on AArch64 systems with TBI mode enabled,
    # and an address mask that clears the top byte before reading
    # from memory.
    @skipUnlessDarwin
    @skipIf(archs=no_match(["arm64", "arm64e"]))
    @skipIfRemote
    def test(self):
        corefile = self.getBuildArtifact("process.core")
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        # Test that we can dereference a pointer with TBI data
        # in a live process.
        self.do_variable_access_tests(thread.GetFrameAtIndex(0))

        # Create a corefile, delete this process
        self.runCmd("process save-core -s stack " + corefile)
        process.Destroy()
        self.dbg.DeleteTarget(target)

        # Now load the corefile
        target = self.dbg.CreateTarget("")
        process = target.LoadCore(corefile)
        thread = process.GetSelectedThread()
        self.assertTrue(process.GetSelectedThread().IsValid())

        # Test that we can dereference a pointer with TBI data
        # in a corefile process.
        self.do_variable_access_tests(thread.GetFrameAtIndex(0))
