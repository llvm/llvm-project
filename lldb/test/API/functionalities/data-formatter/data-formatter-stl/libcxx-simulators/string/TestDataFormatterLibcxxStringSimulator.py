"""
Test we can understand various layouts of the libc++'s std::string
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import functools


class LibcxxStringDataFormatterSimulatorTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def _run_test(self, defines):
        cxxflags_extras = " ".join(["-D%s" % d for d in defines])
        self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxxflags_extras))
        lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp")
        )
        self.expect_var_path("shortstring", summary='"short"')
        self.expect_var_path("longstring", summary='"I am a very long string"')


for v in [None, "ALTERNATE_LAYOUT"]:
    for r in range(5):
        name = "test_r%d" % r
        defines = ["REVISION=%d" % r]
        if v:
            name += "_" + v
            defines += [v]
        f = functools.partialmethod(
            LibcxxStringDataFormatterSimulatorTestCase._run_test, defines
        )
        setattr(LibcxxStringDataFormatterSimulatorTestCase, name, f)
