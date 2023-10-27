"""
Check that LLDB can read Scalable Matrix Extension (SME) data from core files.
"""


import lldb
import itertools
from enum import Enum
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class Mode(Enum):
    SVE = 0
    SSVE = 1


class ZA(Enum):
    Disabled = 0
    Enabled = 1


class AArch64LinuxSMECoreFileTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    # SME introduces an extra SVE mode "streaming mode" and an array storage
    # register "ZA". ZA can be enabled or disabled independent of streaming mode.
    # Vector length can also be different between the streaming and non-streaming
    # mode. Therefore this test checks a few combinations, but not all.
    #
    # The numbers in the core file names are options to the crashing program,
    # see main.c for their meaning. The test case names will also explain them.

    def check_corefile(self, corefile):
        self.runCmd("target create --core " + corefile)

        _, sve_mode, vl, svl, za = corefile.split("_")

        sve_mode = Mode(int(sve_mode))
        vl = int(vl)
        svl = int(svl)
        za = ZA(int(za))

        self.expect("register read tpidr2", substrs=["0x1122334455667788"])

        # In streaming mode, vg is the same as svg. 'g' is for granule which is
        # 8 bytes.
        if sve_mode == Mode.SSVE:
            self.expect("register read vg", substrs=["0x{:016x}".format(svl // 8)])
        else:
            self.expect("register read vg", substrs=["0x{:016x}".format(vl // 8)])

        # svg is always the streaming mode vector length.
        self.expect("register read svg", substrs=["0x{:016x}".format(svl // 8)])

        svcr = 1 if sve_mode == Mode.SSVE else 0
        if za == ZA.Enabled:
            svcr |= 2
        self.expect("register read svcr", substrs=["0x{:016x}".format(svcr)])

        repeat_bytes = lambda v, n: " ".join(["0x{:02x}".format(v)] * n)

        sve_vl = svl if sve_mode == Mode.SSVE else vl
        for i in range(0, 32):
            # Each element is set to the register number + 1, for example:
            #  z0 = {0x01 0x01 0x01 ... }
            expected = "{{{}}}".format(repeat_bytes(i + 1, sve_vl))
            self.expect("register read z{}".format(i), substrs=[expected])

        # The P registers cycle between a few values.
        # p0 = {0xff 0xff ... }
        # p1 = {0x55 0x55 ... }
        # ...
        # P registers and FFR have 1 bit per byte element in a vector.
        p_value = lambda v: "{{{}}}".format(repeat_bytes(v, sve_vl // 8))
        expected_p_values = [p_value(v) for v in [0xFF, 0x55, 0x11, 0x01, 0x00]]
        expected_p_values = itertools.cycle(expected_p_values)

        for i in range(0, 15):
            expected = next(expected_p_values)
            self.expect("register read p{}".format(i), substrs=[expected])

        self.expect(
            "register read ffr",
            substrs=["{{{}}}".format(repeat_bytes(0xFF, sve_vl // 8))],
        )

        if za == ZA.Enabled:
            # Each row of ZA is set to the row number plus 1. For example:
            # za = {0x01 0x01 0x01 0x01 <repeat until end of row> 0x02 0x02 ...
            make_row = repeat_bytes
        else:
            # When ZA is disabled lldb shows it as 0s.
            make_row = lambda _, n: repeat_bytes(0, n)

        expected_za = "{{{}}}".format(
            " ".join([make_row(i + 1, svl) for i in range(svl)])
        )
        self.expect("register read za", substrs=[expected_za])

    @skipIfLLVMTargetMissing("AArch64")
    def test_sme_core_file_ssve_vl32_svl16_za_enabled(self):
        self.check_corefile("core_1_32_16_1")

    @skipIfLLVMTargetMissing("AArch64")
    def test_sme_core_file_ssve_vl16_svl32_za_disabled(self):
        self.check_corefile("core_1_16_32_0")

    @skipIfLLVMTargetMissing("AArch64")
    def test_sme_core_file_sve_vl16_svl32_za_enabled(self):
        self.check_corefile("core_0_16_32_1")

    @skipIfLLVMTargetMissing("AArch64")
    def test_sme_core_file_sve_vl32_svl16_za_disabled(self):
        self.check_corefile("core_0_32_16_0")
