"""
Test DIL basic assignment.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import lldbutil


class TestFrameVarDILAssignment(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_assignment(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        self.expect("frame variable 'i += 1'", substrs=["2"])
        self.expect("frame variable 's += i'", substrs=["9"])
        self.expect("frame variable 'i += 2'", substrs=["4"])
        self.expect("frame variable 'i += -4'", substrs=["0"])
        self.expect("frame variable 'i += eOne'", substrs=["0"])
        self.expect("frame variable 'i += eTwo'", substrs=["1"])

        self.expect("frame variable 'f += 1'", substrs=["2.5"])
        self.expect("frame variable 'f += -2.0f'", substrs=["0.5"])
        self.expect("frame variable 'f += 2.5f'", substrs=["3"])
        self.expect("frame variable 'f += eTwo'", substrs=["4"])

        Is32Bit = False
        if self.target().GetAddressByteSize() == 4:
            Is32Bit = True

        self.expect(
            "frame variable 'i += p'",  # Try assigning pointer to int.
            error=True,
            substrs=["Invalid assignment: Can only assign pointers to pointers"],
        )

        if Is32Bit:
            self.expect("frame variable 'p'", substrs=["p = 0x0000000a"])
            self.expect("frame variable 'p += 2'", substrs=["p = 0x00000012"])
            self.expect("frame variable 'p += i'", substrs=["p = 0x00000016"])
        else:
            self.expect("frame variable 'p'", substrs=["p = 0x000000000000000a"])
            self.expect("frame variable 'p += 2'", substrs=["p = 0x0000000000000012"])
            self.expect("frame variable 'p += i'", substrs=["p = 0x0000000000000016"])

        self.expect("frame variable 'i = 1'", substrs=["1"])
        self.expect("frame variable 'i -= 1'", substrs=["0"])
        self.expect("frame variable 'i -= 2'", substrs=["-2"])
        self.expect("frame variable 'i -= -4'", substrs=["2"])
        self.expect("frame variable 'i -= eOne'", substrs=["2"])
        self.expect("frame variable 'i -= eTwo'", substrs=["1"])

        self.expect("frame variable 'f = 1.5f'", substrs=["1.5"])
        self.expect("frame variable 'f -= 1'", substrs=["0.5"])
        self.expect("frame variable 'f -= -2.0f'", substrs=["2.5"])
        self.expect("frame variable 'f -= -2.5f'", substrs=["5"])
        self.expect("frame variable 'f -= eTwo'", substrs=["4"])

        Is32Bit = False
        if self.target().GetAddressByteSize() == 4:
            Is32Bit = True

        self.expect(
            "frame variable 'i -= p'",  # Try assigning pointer to int.
            error=True,
            substrs=["invalid operands to binary expression"],
        )

        if Is32Bit:
            self.expect("frame variable 'p = (int *)10'", substrs=["p = 0x0000000a"])
            self.expect("frame variable 'p -= 2'", substrs=["p = 0x00000002"])
        else:
            self.expect(
                "frame variable 'p = (int *)10'", substrs=["p = 0x000000000000000a"]
            )
            self.expect("frame variable 'p -= 2'", substrs=["p = 0x0000000000000002"])

        self.expect("frame variable 'i = 2'", substrs=["2"])
        self.expect("frame variable 'i *= 2'", substrs=["4"])
        self.expect("frame variable 'iref *= 2'", substrs=["8"])
        self.expect("frame variable 'd = 2.25'", substrs=["2.25"])
        self.expect("frame variable 'd *= 2.0'", substrs=["4.5"])
        self.expect("frame variable 'd *= 2'", substrs=["9"])
        self.expect(
            "frame variable 'i *= 2.0'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'double' to 'int'"
            ],
        )

        self.expect("frame variable 'i = 7'", substrs=["7"])
        self.expect("frame variable 'i /= 3'", substrs=["2"])
        self.expect("frame variable 'iref /= 2'", substrs=["1"])
        self.expect("frame variable 'f = 4.5f'", substrs=["4.5"])
        self.expect("frame variable 'f /= 2.0f", substrs=["2.25"])
        self.expect("frame variable 'f /= 2", substrs=["1.125"])
        self.expect(
            "frame variable 'i /= 1.0f'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'float' to 'int'"
            ],
        )

        self.expect("frame variable 'i = 7'", substrs=["7"])
        self.expect("frame variable 'i %= 4'", substrs=["3"])
        self.expect("frame variable 'iref %= 3'", substrs=["0"])
        self.expect(
            "frame variable 'i %= 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )

        self.expect("frame variable 'i = 0xF'", substrs=["15"])
        self.expect("frame variable 'i &= 0b101'", substrs=["5"])
        self.expect("frame variable 'iref &= 0b100'", substrs=["4"])
        self.expect(
            "frame variable 'i &= 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )

        self.expect("frame variable 'i = 0xF'", substrs=["15"])
        self.expect("frame variable 'i ^= 0b1000'", substrs=["7"])
        self.expect("frame variable 'iref ^= 0b11'", substrs=["4"])
        self.expect(
            "frame variable 'i ^= 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )

        self.expect("frame variable 'i = 1'", substrs=["1"])
        self.expect("frame variable 'i |= 0b110'", substrs=["7"])
        self.expect("frame variable 'iref |= 0b1001'", substrs=["15"])
        self.expect(
            "frame variable 'i |= 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )

        self.expect("frame variable 'i = 0xF'", substrs=["15"])
        self.expect("frame variable 'i >>= 1'", substrs=["7"])
        self.expect("frame variable 'iref >>= 2'", substrs=["1"])
        self.expect(
            "frame variable 'i >>= 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )

        self.expect("frame variable 'i = 1'", substrs=["1"])
        self.expect("frame variable 'i <<= 1'", substrs=["2"])
        self.expect("frame variable 'iref <<= 2'", substrs=["8"])
        self.expect(
            "frame variable 'i <<= 1.0'",
            error=True,
            substrs=["invalid operands to binary expression ('int' and 'double')"],
        )

        # Check that there can be only one assignment and only at top level
        self.expect(
            "frame variable 'i = i += 1'",
            error=True,
            substrs=["Assignment is allowed only at top level"],
        )
        self.expect(
            "frame variable 'i += 1 + (i -= 1)'",
            error=True,
            substrs=["Assignment is allowed only at top level"],
        )
        self.expect(
            "frame variable '*(arr + 1) = arr[1] += 1'",
            error=True,
            substrs=["Assignment is allowed only at top level"],
        )

        # Check that operators `--` and `++` return an error message.
        self.expect(
            "frame variable 'i++'",
            error=True,
            substrs=["Increment operator is not supported. Use `+=` instead."],
        )
        self.expect(
            "frame variable '++i'",
            error=True,
            substrs=["Increment operator is not supported. Use `+=` instead."],
        )
        self.expect(
            "frame variable 'i--'",
            error=True,
            substrs=["Decrement operator is not supported. Use `-=` instead."],
        )
        self.expect(
            "frame variable -- '--i'",
            error=True,
            substrs=["Decrement operator is not supported. Use `-=` instead."],
        )
