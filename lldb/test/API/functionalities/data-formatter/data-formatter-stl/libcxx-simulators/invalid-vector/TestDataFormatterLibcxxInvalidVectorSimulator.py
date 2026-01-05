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
    def test(self):
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
            substrs=["size=error: size not multiple of element size"],
        )
