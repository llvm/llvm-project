"""
Test we can understand various layouts of the libc++'s std::unique_ptr
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import functools


class LibcxxUniquePtrDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _run_test(self, defines):
        cxxflags_extras = " ".join(["-D%s" % d for d in defines])
        self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxxflags_extras))
        lldbutil.run_to_source_breakpoint(
            self, "Break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect("frame variable var_up", substrs=["pointer ="])
        self.expect("frame variable var_up", substrs=["deleter ="], matching=False)
        self.expect(
            "frame variable var_with_deleter_up", substrs=["pointer =", "deleter ="]
        )


for r in range(3):
    name = "test_r%d" % r
    defines = ["COMPRESSED_PAIR_REV=%d" % r]
    f = functools.partialmethod(
        LibcxxUniquePtrDataFormatterSimulatorTestCase._run_test, defines
    )
    setattr(LibcxxUniquePtrDataFormatterSimulatorTestCase, name, f)
