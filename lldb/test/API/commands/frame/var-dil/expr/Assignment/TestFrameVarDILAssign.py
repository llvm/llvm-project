"""
Test DIL basic assignment.
"""

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
from lldbsuite.test import *


class TestFrameVarDILAssignment(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test_assignment(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set a breakpoint here", lldb.SBFileSpec("main.cpp")
        )

        self.runCmd("settings set target.experimental.use-DIL true")

        Is32Bit = False
        if self.target().GetAddressByteSize() == 4:
            Is32Bit = True

        self.expect(
            "frame variable '1 = 1'",
            error=True,
            substrs=["Not allowed to change the value of a constant"],
        )

        # Assigning to an int var
        self.expect_var_path("i", value="1")
        self.expect("frame variable 'i = 5'", substrs=["i = 5"])
        self.expect_var_path("i", value="5")
        self.expect_var_path("j", value="-4")
        self.expect("frame variable 'i = j'", substrs=["i = -4"])
        self.expect_var_path("i", value="-4")
        self.expect("frame variable 'i = 2'", substrs=["i = 2"])
        self.expect_var_path("i", value="2")
        self.expect("frame variable 'i = -2'", substrs=["i = -2"])
        self.expect_var_path("i", value="-2")
        self.expect("frame variable 'i = (int)eOne'", substrs=["i = 0"])
        self.expect_var_path("i", value="0")
        self.expect("frame variable 'i = (int)eTwo'", substrs=["i = 1"])
        self.expect_var_path("i", value="1")

        # Assigning to a float var
        self.expect_var_path("f", value="1.5")
        self.expect("frame variable 'f = 17.823f'", substrs=["f = 17.823"])
        self.expect_var_path("f", value="17.823")
        self.expect_var_path("pi", value="3.14159012")
        self.expect("frame variable 'f = pi'", substrs=["f = 3.14159012"])
        self.expect_var_path("f", value="3.14159012")
        self.expect("frame variable 'f = 2.5f'", substrs=["f = 2.5"])
        self.expect_var_path("f", value="2.5")
        self.expect("frame variable 'f = 3.5f'", substrs=["f = 3.5"])
        self.expect_var_path("f", value="3.5")

        # Assigning to an enum
        self.expect("frame variable 'i = eOne'", substrs=["0"])
        self.expect("frame variable 'eOne = 1'", substrs=["eOne = TWO"])
        self.expect("frame variable 'eOne = 0'", substrs=["eOne = ONE"])

        # Assigning to a pointer
        self.expect(
            "frame variable 'p = 1'",
            error=True,
            substrs=["Invalid assignment: Can only assign pointers to pointers"],
        )

        self.expect(
            "frame variable 'p = i + s'",
            error=True,
            substrs=["Invalid assignment: Can only assign pointers to pointers"],
        )

        self.expect(
            "frame variable 'i = p'",
            error=True,
            substrs=["Invalid assignment: Can only assign pointers to pointers"],
        )

        if Is32Bit:
            self.expect("frame variable 'p = (int*)12'", substrs=["p = 0x0000000c"])
            self.expect_var_path("p", value="0x0000000c")
            self.expect("frame variable 'p = p - s'", substrs=["p = 0x0000000b"])
            self.expect("frame variable 'p = (int *)0'", substrs=["p = 0x00000000"])
        else:
            self.expect(
                "frame variable 'p = (int*)12'", substrs=["p = 0x000000000000000c"]
            )
            self.expect_var_path("p", value="0x000000000000000c")
            self.expect(
                "frame variable 'p = (int *)0'", substrs=["p = 0x0000000000000000"]
            )

        # Assigning to a bool
        self.expect_var_path("b", value="false")
        self.expect("frame variable 'b = true'", substrs=["b = true"])
        self.expect_var_path("b", value="true")
        self.expect_var_path("(int)b", value="")
        self.expect("frame variable 'b = (bool)0'", substrs=["b = false"])
        self.expect_var_path("b", value="false")

        # Assigning to an array
        self.expect("frame variable 'farr'", substrs=["([0] = 1, [1] = 2)"])
        self.expect("frame variable 'farr[1] = f'", substrs=["farr[1] = f = 3.5"])
        self.expect_var_path("farr[1]", value="3.5")
        self.expect("frame variable 'farr'", substrs=["([0] = 1, [1] = 3.5)"])
        self.expect("frame variable 'arr'", substrs=["([0] = 1, [1] = 2)"])
        self.expect("frame variable 'arr[0] = 37'", substrs=["arr[0] = 37"])
        self.expect("frame variable 'arr[1] = j'", substrs=["arr[1] = j = -4"])
        self.expect("frame variable 'arr'", substrs=["([0] = 37, [1] = -4)"])

        # Test basic assignment conversions.
        # int = enum
        self.expect("frame variable 'i = eOne'", substrs=["i = 0"])
        # int = bool
        self.expect("frame variable 'i = true'", substrs=["i = 1"])
        # int = double
        self.expect(
            "frame variable 'i = d2'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'double' to 'int'"
            ],
        )
        # int = float
        self.expect(
            "frame variable 'i = pi'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'float' to 'int'"
            ],
        )
        # float = int
        self.expect(
            "frame variable 'f = 8'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'int' to 'float'"
            ],
        )
        # float = double
        self.expect(
            "frame variable 'f = d2'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'double' to 'float'"
            ],
        )
        # double = double
        self.expect("frame variable 'd = d2'", substrs=["d = 15.779999999999999"])
        self.expect("frame variable 'd = 1.25'", substrs=["d = 1.25"])
        # double = float
        self.expect(
            "frame variable 'd = pi'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'float' to 'double'"
            ],
        )
        # double = int
        self.expect(
            "frame variable 'd = 17'",
            error=True,
            substrs=[
                "Incompatible types for assignment: Cannot assign 'int' to 'double'"
            ],
        )

        # bool = int
        self.expect("frame variable 'b = 0'", substrs=["b = false"])
        self.expect("frame variable 'b = 1'", substrs=["b = true"])
        self.expect("frame variable 'b = 32'", substrs=["b = true"])

        # Assigning "int" with a small value to "short" should work.
        self.expect("frame variable 's = i'", substrs=["s = 1"])
        # Assigning "int" with a big value to "short" should fail.
        self.expect(
            "frame variable 's = 78246'",
            error=True,
            substrs=["new value is too big"],
        )
