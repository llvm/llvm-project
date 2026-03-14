import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestArm64eAttach(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # On Darwin systems, arch arm64e means ARMv8.3 with ptrauth ABI used.
    @skipIf(archs=no_match(["arm64e"]))
    def test(self):
        # Skip this test if not running on AArch64 target that supports PAC
        if not self.isAArch64PAuth():
            self.skipTest("Target must support pointer authentication.")

        self.build()
        popen = self.spawnSubprocess(self.getBuildArtifact(), [])

        # This simulates how Xcode attaches to a process by pid/name.
        error = lldb.SBError()
        target = self.dbg.CreateTarget("", "arm64", "", True, error)
        listener = lldb.SBListener("my.attach.listener")
        process = target.AttachToProcessWithID(listener, popen.pid, error)
        self.assertSuccess(error)
        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(target.GetTriple().split('-')[0], "arm64e",
                         "target triple is updated correctly")

        self.expect('process plugin packet send qProcessInfo',
                    "debugserver returns correct triple",
                    substrs=['cputype:100000c', 'cpusubtype:2', 'ptrsize:8'])

        error = process.Kill()
        self.assertSuccess(error)
