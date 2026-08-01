"""Check that enumerator values are correct when they're near the limits of the underlying type."""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CPPEnumLimitsTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    TEST_WITH_PDB_DEBUG_INFO = True

    def check_signed(self, ty: lldb.SBType, expected):
        self.assertEqual(
            {i.GetName(): i.GetValueAsSigned() for i in ty.GetEnumMembers()}, expected
        )

    def check_unsigned(self, ty: lldb.SBType, expected):
        self.assertEqual(
            {i.GetName(): i.GetValueAsUnsigned() for i in ty.GetEnumMembers()}, expected
        )

    def check_all(self, target: lldb.SBTarget):
        self.check_unsigned(target.FindFirstType("U8Enum"), {"Min": 0, "Max": 255})
        self.check_signed(target.FindFirstType("I8Enum"), {"Min": -128, "Max": 127})
        self.check_unsigned(target.FindFirstType("U16Enum"), {"Min": 0, "Max": 65535})
        self.check_signed(
            target.FindFirstType("I16Enum"), {"Min": -32768, "Max": 32767}
        )
        self.check_unsigned(
            target.FindFirstType("U32Enum"),
            {"Min": 0, "MaxMinusOne": 4294967294, "Max": 4294967295},
        )
        self.check_signed(
            target.FindFirstType("I32Enum"), {"Min": -2147483648, "Max": 2147483647}
        )
        self.check_unsigned(
            target.FindFirstType("U64Enum"),
            {
                "Min": 0,
                "MaxMinusOne": 18446744073709551614,
                "Max": 18446744073709551615,
            },
        )
        self.check_signed(
            target.FindFirstType("I64Enum"),
            {
                "Min": -9223372036854775808,
                "MinPlusOne": -9223372036854775807,
                "MaxMinusOne": 9223372036854775806,
                "Max": 9223372036854775807,
            },
        )

    def test(self):
        self.build()
        self.check_all(self.dbg.CreateTarget(self.getBuildArtifact("a.out")))

    @skipUnlessPlatform(["windows"])
    @skipUnlessMSVC
    @no_debug_info_test  # We only test MSVC
    def test_msvc(self):
        """Test that the limits work on MSVC."""

        src = os.path.join(self.getSourceDir(), "main.cpp")
        exe = os.path.join(self.getBuildDir(), "a.exe")

        # FIXME: Allow MSVC to be used with the Makefile.
        result = subprocess.run(
            [
                "cl.exe",
                "/nologo",
                "/Od",
                "/Zi",
                "/Fe" + exe,
                src,
                "/link",
                "/nodefaultlib",
                "/entry:main",
            ],
            cwd=self.getBuildDir(),
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            "Compilation failed:\n" + result.stdout + result.stderr,
        )

        self.check_all(self.dbg.CreateTarget(exe))
