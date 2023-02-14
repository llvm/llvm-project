"""
Test that "breakpoint set -a" uses the ABI plugin to remove non-address bits
before attempting to set a breakpoint.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxNonAddressBitCodeBreak(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def do_tagged_break(self, hardware):
        if not self.isAArch64PAuth():
            self.skipTest('Target must support pointer authentication.')

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.c",
            line_number('main.c', '// Set break point at this line.'),
            num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped',
                     'stop reason = breakpoint'])

        cmd = "breakpoint set -a fnptr"
        # LLDB only has the option to force hardware break, not software.
        # It prefers sofware break when it can and this will be one of those cases.
        if hardware:
            cmd += " --hardware"
        self.expect(cmd)

        self.runCmd("continue")
        self.assertEqual(self.process().GetState(), lldb.eStateStopped)
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', '`foo at main.c', 'stop reason = breakpoint'])

    # AArch64 Linux always enables the top byte ignore feature
    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_software_break(self):
        self.do_tagged_break(False)

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_hardware_break(self):
        self.do_tagged_break(True)
