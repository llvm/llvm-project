"""
Test AArch64 SVE registers on Windows.

NOTE: The default non-streaming SVE vector length is 128 bits, so Z register
reads and writes currently go through the V register read and write paths. As a
result, the interleaving of the low and high bits of Z registers is NOT
exercised. P/FFR read and P write paths ARE exercised.

TODO: Add coverage for a wider vector length.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.cpu_feature as cpu_feature


class SVETestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # main()
        self.line1 = line_number("main.c", "// breakpoint 1")

    def run_sve_test(self):
        # Set breakpoints
        self.runCmd("breakpoint set -f main.c -l " + str(self.line1))

        # Run the program.
        self.runCmd("run", RUN_SUCCEEDED)

        # Stopped at 'breakpoint 1', with all SVE registers set.

        # The default non-streaming mode vector length is 128 bits (16 bytes).
        byte_length = 16

        # Load 'zregs' with a byte pattern consisting of register number
        # followed by 0-7:
        # 00 00 00 01 00 02 00 03 ... 00 07
        # 01 00 01 01 01 02 01 03 ... 01 07
        # ...
        # 31 00 31 01 31 02 31 03 ... 31 07
        zregs = ["" for i in range(32)]

        for i in range(32):
            bytes_list = []
            for j in range(byte_length // 2):
                bytes_list.append("0x%02x" % i)
                bytes_list.append("0x%02x" % j)
            zregs[i] = "{" + " ".join(bytes_list) + "}"

        # Load 'pregs' with a byte pattern consisting of register number
        # followed by 0:
        # 00 00
        # 01 00
        # ...
        # 31 00
        pregs = ["" for i in range(16)]

        for i in range(16):
            bytes_list = []
            for j in range(byte_length // 16):
                bytes_list.append("0x%02x" % i)
                bytes_list.append("0x%02x" % j)
            pregs[i] = "{" + " ".join(bytes_list) + "}"

        # load 'ffr' bytes with a byte pattern consisting of all 1s:
        # ff ff
        bytes_list = []
        for i in range(byte_length // 8):
            bytes_list.append("0xff")
        ffr = "{" + " ".join(bytes_list) + "}"

        # Test that 'vg' has the correct value.
        self.expect(
            "register read vg",
            "vg is correct",
            substrs=["vg = 0x00000000000000%02x" % (byte_length // 8)],
        )

        # Test that Z registers have the correct values.
        for i in range(32):
            self.expect(
                "register read z" + str(i),
                "sve register z" + str(i) + " is valid",
                substrs=["z" + str(i) + " = " + zregs[i]],
            )

        # Test that P registers have the correct values.
        for i in range(16):
            self.expect(
                "register read p" + str(i),
                "sve register p" + str(i) + " is valid",
                substrs=["p" + str(i) + " = " + pregs[i]],
            )

        # Test that FFR has the correct value.
        self.expect(
            "register read ffr",
            "sve register ffr is valid",
            substrs=["ffr" + " = " + ffr],
        )

        # Write 'z7', then read it back.
        # 'z7' has value '0x07000701...'. Write it with something different, and
        # check it. The saved value should be the value that we wrote.
        bytes_pattern = [
            "00",
            "11",
            "22",
            "33",
            "44",
            "55",
            "66",
            "77",
            "88",
            "99",
            "aa",
            "bb",
            "cc",
            "dd",
            "ee",
            "ff",
        ]

        bytes_list = []
        bytes_list.extend("0x" + b for b in bytes_pattern)
        my_z7 = "{" + " ".join(bytes_list) + "}"

        self.runCmd("register write z7 '" + my_z7 + "'")
        self.expect(
            "register read z7",
            "z7 has correct value after write",
            substrs=["z7 = " + my_z7],
        )

        # Write 'p7', then read it back.
        # 'p7' has value '0x0700'. Write it with something different, and check
        # it. The saved value should be the value that we wrote.
        bytes_pattern = [
            "00",
            "11",
        ]

        bytes_list = []
        bytes_list.extend("0x" + b for b in bytes_pattern)
        my_p7 = "{" + " ".join(bytes_list) + "}"

        self.runCmd("register write p7 '" + my_p7 + "'")
        self.expect(
            "register read p7",
            "p7 has correct value after write",
            substrs=["p7 = " + my_p7],
        )

    # Currently, the test is supported only on WoA devices with SVE support when
    # run through lldb-server and when the debugger is built with Windows SDK
    # 10.0.26100 (Windows 11 24H2) or later. One caveat, however, is that if the
    # debugger was built with an older Windows SDK, the test will still run on a
    # WoA device with SVE support through lldb-server but it will fail.
    @skipUnlessWindows
    @skipUnlessFeature(cpu_feature.AArch64.SVE)
    @skipIf(remote=False)
    def test_sve(self):
        """Test SVE register access, non-streaming"""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)
        self.run_sve_test()
