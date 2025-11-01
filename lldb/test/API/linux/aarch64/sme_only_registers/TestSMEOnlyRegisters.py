"""
Check handling of registers on an AArch64 Linux system that only has SME. As
opposed to SVE and SME. Check register access and restoration after expression
evaluation.
"""

from enum import Enum
from pprint import pprint
from functools import lru_cache
from itertools import permutations
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from itertools import cycle


class Mode(Enum):
    SIMD = 0
    SSVE = 2

    def __str__(self):
        return "streaming" if self == Mode.SSVE else "simd"


class ZA(Enum):
    ON = 1
    OFF = 2

    def __str__(self):
        return "on" if self == ZA.ON else "off"


class ByteVector(object):
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return "{" + " ".join([f"0x{b:02x}" for b in self.data]) + "}"

    def as_bytes(self):
        return self.data


class HexValue(object):
    # Assume all values are 64-bit, but some display as 32-bit, like fpcr.
    def __init__(self, value, repr_size=8):
        self.value = value
        # In bytes
        self.repr_size = repr_size

    def __repr__(self):
        return f"0x{self.value:0{self.repr_size*2}x}"

    def as_bytes(self):
        data = []
        v = self.value
        # Little endian order.
        for i in range(8):
            data.append(v & 0xFF)
            v >>= 8

        return data


class SVESIMDRegistersTestCase(TestBase):
    def reg_names(self, prefix, count):
        return [f"{prefix}{n}" for n in range(count)]

    def expected_registers(self, svl_b, mode, za):
        register_values = []

        if mode == Mode.SIMD:
            # In streaming mode we have real V registers and no Z registers.

            # V regs are {N <7 0s> N <7 0s>} because we set the bottom element to N
            # where N is 1 + the register index.
            v_values = [
                ByteVector([n + 1] + [0] * 7 + [n + 1] + [0] * 7) for n in range(32)
            ]

            # Z regs are {N <7 0s> N <7 0s> <16 more 0s}. First half overlaps a V
            # register, the second half we fake 0s for as there is no real Z register
            # in non-streaming mode.
            z_values = [
                ByteVector([n + 1] + [0] * 7 + [n + 1] + [0] * 7 + [0] * (svl_b - 16))
                for n in range(32)
            ]

            # P regs are {<4 0s>}, we fake the value.
            p_values = [ByteVector([0] * (svl_b // 8)) for _ in range(16)]
        else:
            # In streaming mode, Z registers are real and V are the bottom 128
            # bits of the Z registers.

            # Streaming SVE registers have their elements set to their number plus 1.
            # So z0 has elements of 0x01, z1 is 0x02 and so on.
            v_values = [ByteVector([n + 1] * 16) for n in range(32)]

            z_values = [ByteVector([n + 1] * svl_b) for n in range(32)]

            # P registers have all emlements set to the same value and that value
            # cycles between 0xff, 0x55, 0x11, 0x01 and 0x00.
            p_values = []
            for i, v in zip(range(16), cycle([0xFF, 0x55, 0x11, 0x01, 0x00])):
                p_values.append(ByteVector([v] * (svl_b // 8)))

        # Would use strict=True here but it requires Python 3.10.
        register_values += list(zip(self.reg_names("v", 32), v_values))
        register_values += [
            ("fpsr", HexValue(0x50000015, repr_size=4)),
            ("fpcr", HexValue(0x05551505, repr_size=4)),
        ]
        register_values += list(zip(self.reg_names("z", 32), z_values))
        register_values += list(zip(self.reg_names("p", 16), p_values))

        # ffr is all 0s. In SIMD mode we're faking the value, in streaming mode,
        # use of ffr is illegal so the kernel tells us it's 0s.
        register_values += [("ffr", ByteVector([0] * (svl_b // 8)))]

        svcr_value = 1 if mode == Mode.SSVE else 0
        if za == ZA.ON:
            svcr_value += 2

        register_values += [
            ("svcr", HexValue(svcr_value)),
            # SVG is the streaming vector length in granules.
            ("svg", HexValue(svl_b // 8)),
        ]

        # ZA and ZTO may be enabled or disabled regardless of streaming mode.
        # ZA is a square of vector length * vector length.
        # ZT0 is 512 bits regardless of streaming vector length.
        if za == ZA.ON:
            register_values += [("za", ByteVector(list(range(1, svl_b + 1)) * svl_b))]
            if self.isAArch64SME2():
                register_values += [("zt0", ByteVector(list(range(1, (512 // 8) + 1))))]
        else:
            # ZA is fake.
            register_values += [("za", ByteVector([0x0] * (svl_b * svl_b)))]
            # ZT0 is also fake.
            if self.isAArch64SME2():
                register_values += [("zt0", ByteVector([0x00] * (512 // 8)))]

        return dict(register_values)

    def check_expected_regs_fn(self, expected_registers):
        def check_expected_regs():
            self.expect(
                f'register read {" ".join(expected_registers.keys())}',
                substrs=[f"{n} = {v}" for n, v in expected_registers.items()],
            )

        return check_expected_regs

    def skip_if_not_sme_only(self):
        if self.isAArch64SVE():
            self.skipTest("SVE must not be present outside of streaming mode.")

        if not self.isAArch64SME():
            self.skipTest("SSVE registers must be supported.")

    def setup_test(self, mode, za, svl):
        self.build()
        self.line = line_number("main.c", "// Set a break point here.")

        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        self.runCmd(f"settings set target.run-args {mode} {za} {svl}")

        lldbutil.run_break_set_by_file_and_line(
            self, "main.c", self.line, num_expected_locations=1
        )
        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread backtrace",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stop reason = breakpoint 1."],
        )

    def write_expected_reg_data(self, reg_data):
        # Write expected register values into program memory so it can be
        # verified in-process.
        # This must be done via. memory write instead of expressions because
        # the latter may try to save/restore registers, which is part of what
        # this file tests so we can't rely on it here.
        # We will always write Z and ZA/ZTO, it's up to the program whether it
        # checks them.

        for reg, value in reg_data.items():
            sym_name = None
            # Since we cannot expression evaluate, we have to manually offset
            # arrays.
            offset = 0

            # Offsets for scalable registers assume that the expected register
            # data length matches the current svl.
            if reg == "fpcr":
                sym_name = "expected_fpcr"
            elif reg == "fpsr":
                sym_name = "expected_fpsr"
            elif reg == "ffr":
                sym_name = "expected_sve_ffr"
            elif reg == "za":
                sym_name = "expected_za"
            elif reg == "zt0":
                sym_name = "expected_zt0"
            elif reg == "svcr":
                sym_name = "expected_svcr"
            elif reg == "svg":
                sym_name = "expected_svg"
            elif reg.startswith("v"):
                num = int(reg.split("v")[1])
                offset = 16 * num
                sym_name = "expected_v_regs"
            elif reg.startswith("z"):
                num = int(reg.split("z")[1])
                offset = len(value.as_bytes()) * num
                sym_name = "expected_sve_z"
            elif reg.startswith("p"):
                num = int(reg.split("p")[1])
                offset = len(value.as_bytes()) * num
                sym_name = "expected_sve_p"

            if sym_name is None:
                raise RuntimeError(
                    f"Do not know how to write expected value for register {reg}."
                )

            address = self.lookup_address(sym_name) + offset
            process = self.dbg.GetSelectedTarget().GetProcess()
            err = lldb.SBError()
            wrote = process.WriteMemory(address, bytearray(value.as_bytes()), err)
            self.assertTrue(err.Success())
            self.assertEqual(len(value.as_bytes()), wrote)

    # This is safe to cache because each test will be its own instance of this
    # class.
    @lru_cache
    def lookup_address(self, sym_name):
        target = self.dbg.GetSelectedTarget()
        sym = target.module[0].FindSymbol(sym_name)
        self.assertTrue(sym.IsValid())
        address = sym.GetStartAddress().GetLoadAddress(target)

        # Dereference the pointer.
        err = lldb.SBError()
        ptr = target.GetProcess().ReadPointerFromMemory(address, err)
        self.assertTrue(err.Success())

        return ptr

    def get_svls(self):
        # This proc file contains the default streaming vector length (in bytes).
        # It defaults to 32, or the largest vector length. Whichever is smaller.
        err, retcode, output = self.run_platform_command(
            "cat /proc/sys/abi/sme_default_vector_length"
        )
        if err.Fail() or retcode != 0:
            self.skipTest(f"Failed to read sme_default_vector_length: {output}")

        # Content should be a single decimal number.
        try:
            default_svl = int(output)
        except ValueError:
            # File contained unexpected data.
            self.skipTest(
                f"sme_default_vector_length contained unexpected data: {output}"
            )

        # A valid svl is a multiple of 128 bits and a power of 2. We need to find
        # 2 of them that will work. We could try to set this file to a higher value,
        # but likely we need root permissions on some machines.
        # So if the default is 128 bit, assume that's the max, in which case
        # there is no second svl we can use.
        if default_svl == 16:
            self.skipTest(
                f"Did not find 2 supported streaming vector lengths, default is {default_svl}"
            )

        # The default is something greater than 16. Use it and the next lowest.
        return (default_svl, default_svl // 2)

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_simd_registers_ssve(self):
        self.skip_if_not_sme_only()

        svl_b = self.get_svls()[0]
        self.setup_test(Mode.SSVE, ZA.ON, svl_b)

        expected_registers = self.expected_registers(svl_b, Mode.SSVE, ZA.ON)
        check_expected_regs = self.check_expected_regs_fn(expected_registers)

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        # Write via Z0
        z_value = ByteVector([0x12] * svl_b)
        self.runCmd(f'register write z0 "{z_value}"')

        # z0 and v0 should change but nothing else.
        expected_registers["z0"] = z_value
        expected_registers["v0"] = ByteVector([0x12] * 16)

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        # We can do the same via a V register, the value will be extended and sent as
        # a Z write.
        v_value = ByteVector([0x34] * 16)
        self.runCmd(f'register write v1 "{v_value}"')

        # The lower half of z1 is the v value, the upper part is the 0x2 that was previously in there.
        expected_registers["z1"] = ByteVector([0x34] * 16 + [0x02] * (svl_b - 16))
        expected_registers["v1"] = v_value

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        # Even though you can't set all these bits in reality, until we do
        # a step, it'll seem like we did.
        # This arbitrary value is 0x55...55 when written to the real register.
        # Some bits cannot be set.
        fpsr = HexValue(0xA800008A, repr_size=4)

        self.runCmd(f"register write fpsr {fpsr}")
        expected_registers["fpsr"] = fpsr

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        # Again this is 0x55...55, but with bits we cannot set removed.
        fpcr = HexValue(0x05551505, repr_size=4)
        self.runCmd(f"register write fpcr {fpcr}")
        expected_registers["fpcr"] = fpcr

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        p_value = ByteVector([0x65] * (svl_b // 8))
        self.expect(f'register write p0 "{p_value}"')
        expected_registers["p0"] = p_value

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        # We cannot interact with ffr in streaming mode while in process. So this
        # will be verified by ptrace only.
        ffr_value = ByteVector([0x78] * (svl_b // 8))
        self.expect(f'register write ffr "{ffr_value}"')
        expected_registers["ffr"] = ffr_value

        # It will appear as if we wrote ffr, but in streaming mode without
        # SME_FA64+SVE, it essentially does not exist.
        check_expected_regs()

        # At least make sure we didn't disturb anything else.
        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])

        # The kernel will always return 0s for ffr.
        expected_registers["ffr"] = ByteVector([0x0] * (svl_b // 8))
        check_expected_regs()

        za_value = ByteVector(list(range(2, svl_b + 2)) * svl_b)
        self.expect(f'register write za "{za_value}"')
        expected_registers["za"] = za_value

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

        # ZT0 is 512 bit regardless of vector length.
        if self.isAArch64SME2():
            zt0_value = ByteVector(list(range(2, (512 // 8) + 2)))
            self.expect(f'register write zt0 "{zt0_value}"')
            expected_registers["zt0"] = zt0_value

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])
        check_expected_regs()

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_simd_registers_simd(self):
        self.skip_if_not_sme_only()

        svl_b = self.get_svls()[0]
        self.setup_test(Mode.SIMD, ZA.OFF, svl_b)

        # Check for the values the program should have set.
        expected_registers = self.expected_registers(svl_b, Mode.SIMD, ZA.OFF)
        check_expected_regs = self.check_expected_regs_fn(expected_registers)

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])

        check_expected_regs()

        # In SIMD mode if you write Z0, only the parts that overlap V0 will
        # change.
        z_value = ByteVector([0x12] * svl_b)
        self.runCmd(f'register write z0 "{z_value}"')

        # z0 and z0 should change but nothing else. We check the rest because
        # we are faking Z register data in this mode, and any offset mistake
        # could lead to modifying other registers.
        expected_registers["z0"] = ByteVector([0x12] * 16 + [0x00] * (svl_b - 16))
        expected_registers["v0"] = ByteVector([0x12] * 16)

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])

        check_expected_regs()

        # We can do the same via a V register, the value will be extended and sent as
        # a Z write.
        v_value = ByteVector([0x34] * 16)
        self.runCmd(f'register write v1 "{v_value}"')

        expected_registers["z1"] = ByteVector([0x34] * 16 + [0x00] * (svl_b - 16))
        expected_registers["v1"] = v_value

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])

        check_expected_regs()

        # FPSR and FPCR are still described as real registers, so they are
        # sent as normal writes.
        # This is the value 0xaaaaaaaa but only the bits that we can actually
        # set in reality.
        fpcontrol = 0xA800008A

        # First FPSR on its own.
        self.runCmd(f"register write fpsr 0x{fpcontrol:08x}")
        expected_registers["fpsr"] = HexValue(fpcontrol, repr_size=4)

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])

        check_expected_regs()

        # Then FPCR. This value is 0xaaaaaaaa reduced to the bits we can actually
        # set.
        fpcontrol = 0x02AAAA02
        self.runCmd(f"register write fpcr 0x{fpcontrol:08x}")
        expected_registers["fpcr"] = HexValue(fpcontrol, repr_size=4)

        self.write_expected_reg_data(expected_registers)
        self.expect("next", substrs=["stop reason = step over"])

        check_expected_regs()

        # We are faking SVE registers while outside of streaming mode, and
        # predicate registers and ffr have no real register to overlay.
        # We chose to make this an error instead of eating the write silently.

        value = ByteVector([0x98] * (svl_b // 8))
        self.expect(f'register write p0 "{value}"', error=True)
        check_expected_regs()
        self.expect(f'register write ffr "{value}"', error=True)
        check_expected_regs()

        # In theory we could test writing to ZA and ZT0, however this would
        # enable streaming mode. In streaming mode, their handling is the same
        # as on an SVE+SME system, and so is covered in other tests.

    @no_debug_info_test
    @skipIf(archs=no_match(["aarch64"]))
    @skipIf(oslist=no_match(["linux"]))
    def test_expr_restore(self):
        """
        Check that we can restore an intial state after expression evaluation
        leaves us in a different state.
        """
        self.skip_if_not_sme_only()

        # Each test goes from a start state to an expression state, and
        # back to the start state. Those state contains:
        # * Streaming mode on or off
        # * ZA on or off
        # * Streaming vector length
        #
        # Rather than trying to be clever choosing which transitions to test,
        # test all the combinations where the start and end state are different.
        #
        # In theory a CPU can support many vector lengths, but the key difficulty
        # with vector length is resizing buffers in LLDB. So we will test 1 "large"
        # length (the default length) and one "small" length (the next smallest
        # than the default). This will cover increasing and decreasing register
        # size.
        #
        # Note that vector length applies to Z and to ZA/ZT0. Even if streaming
        # mode is not enabled, ZA/ZT0 can change size.
        #
        # These tests take a very long time and in theory we could do them not
        # by re-running the program but by changing the state via. register
        # writes. However, some states cannot be accessed by writes done by LLDB
        # so to keep things simple we handle all states the same way.
        #
        # Doing it this way also gives us extra register reading coverage in all
        # the possible states.
        states = []
        svls = self.get_svls()
        for m in list(Mode):
            for za in list(ZA):
                for vl in svls:
                    states.append((m, za, vl))

        # Start with not changing state.
        expr_tests = [(state, state) for state in states]
        # Then all combinations of different states.
        expr_tests.extend(list(permutations(states, 2)))

        if self.TraceOn():
            print("Expression tests:")
            pprint(expr_tests)

        for (sm, sz, svl), (em, ez, evl) in expr_tests:
            if self.TraceOn():
                print(
                    f"Testing restore to [mode:{sm} za:{sz} svl:{svl}] from expression state [mode:{em} za:{ez} svl:{evl}]"
                )

            self.setup_test(sm, sz, svl)

            expected_registers = self.expected_registers(svl, sm, sz)
            check_expected_regs = self.check_expected_regs_fn(expected_registers)

            # The program sets up the initial state by running code in process.
            # In theory we could skip this, but it does give us coverage of
            # reading registers in all modes.
            check_expected_regs()
            # This expression will change modes and set different values.
            self.expect(
                f"expression expr_function({str(em == Mode.SSVE).lower()}, {str(ez == ZA.ON).lower()}, {evl})"
            )
            # LLDB should restore the process to the previous values and modes.
            check_expected_regs()

            self.runCmd("process kill")
