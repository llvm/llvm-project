"""Test that the expedited thread pc values are not re-fetched by lldb."""

import subprocess
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

file_index = 0


class TestExpeditedThreadPCs(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    def test_expedited_thread_pcs(self):
        TestBase.setUp(self)

        global file_index
        ++file_index
        logfile = os.path.join(
            self.getBuildDir(),
            "packet-log-" + self.getArchitecture() + "-" + str(file_index) + ".txt",
        )
        self.runCmd("log enable -f %s gdb-remote packets" % (logfile))

        def cleanup():
            self.runCmd("log disable gdb-remote packets")
            if os.path.exists(logfile):
                os.unlink(logfile)

        self.addTearDownHook(cleanup)

        self.source = "main.cpp"
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec(self.source, False)
        )

        # verify that libfoo.dylib hasn't loaded yet
        for m in target.modules:
            self.assertNotEqual(m.GetFileSpec().GetFilename(), "libfoo.dylib")

        thread.StepInto()
        thread.StepInto()

        thread.StepInto()
        thread.StepInto()
        thread.StepInto()

        # verify that libfoo.dylib has loaded
        for m in target.modules:
            if m.GetFileSpec().GetFilename() == "libfoo.dylib":
                found_libfoo = True
        self.assertTrue(found_libfoo)

        thread.StepInto()
        thread.StepInto()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()
        thread.StepOver()

        process.Kill()

        # Confirm that we never fetched the pc for any threads during
        # this debug session.
        if os.path.exists(logfile):
            f = open(logfile)
            lines = f.readlines()
            num_errors = 0
            for line in lines:
                arch = self.getArchitecture()
                if arch == "arm64" or arch == "arm64_32":
                    #   <reg name="pc" regnum="32" offset="256" bitsize="64" group="general" group_id="1" ehframe_regnum="32" dwarf_regnum="32" generic="pc"/>
                    # A fetch of $pc on arm64 looks like
                    #  <  22> send packet: $p20;thread:91698e;#70
                    self.assertNotIn("$p20;thread", line)
                else:
                    #   <reg name="rip" regnum="16" offset="128" bitsize="64" group="general" altname="pc" group_id="1" ehframe_regnum="16" dwarf_regnum="16" generic="pc"/>
                    # A fetch of $pc on x86_64 looks like
                    #  <  22> send packet: $p10;thread:91889c;#6f
                    self.assertNotIn("$p10;thread", line)

            f.close()
