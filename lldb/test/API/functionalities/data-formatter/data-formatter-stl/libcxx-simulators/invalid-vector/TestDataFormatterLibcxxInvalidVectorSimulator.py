"""
Test we can understand various layouts of the libc++'s std::string
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import functools


class LibcxxInvalidVectorDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(compiler="clang", compiler_version=['<', '18.0'])
    def test_most(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))

        self.expect(
            "frame variable v1",
            substrs=["size=error: invalid value for end of vector"],
        )
        self.expect(
            "frame variable v2",
            substrs=["size=error: invalid value for start of vector"],
        )
        self.expect(
            "frame variable v3",
            substrs=["size=error: start of vector data begins after end pointer"],
        )
        self.expect(
            "frame variable v4",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v5",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v6",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v7",
            substrs=["size=error: invalid value for end of vector"],
        )
        self.expect(
            "frame variable v8",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v9",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v10",
            substrs=["size=error: invalid value for end of vector"],
        )
        self.expect(
            "frame variable v11",
            substrs=["size=error: invalid value for start of vector"],
        )
        self.expect(
            "frame variable v12",
            substrs=["size=error: start of vector data begins after end pointer"],
        )
        self.expect(
            "frame variable v13",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v14",
            substrs=["size=error: invalid value for end of vector"],
        )
        self.expect(
            "frame variable v15",
            substrs=["size=1"],
        )
        self.expect(
            "frame variable v16",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v17",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v18",
            substrs=["size=error: size data member must be a built-in integer type"],
        )
        self.expect(
            "frame variable v19",
            substrs=["size=error: size not multiple of element size"],
        )
        self.expect(
            "frame variable v20",
            substrs=["size=error: size not multiple of element size"],
        )
        self.expect(
            "frame variable v21",
            substrs=["size=1"],
        )

    @skipIf(compiler="clang", compiler_version=["<", "18.0"])
    @skipIfWindows
    def test_zero_sized_struct_extension(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))

        self.expect(
            "frame variable v23",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v24",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
        self.expect(
            "frame variable v25",
            substrs=["size=error: failed to determine start/end of vector data"],
        )
