"""
Tests const static data members as specified by C++11 [class.static.data]p3
with (u)int128_t types.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test_int128(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
                                          lldb.SBFileSpec("main.cpp"))

        # Try to use the (u)int128_t data members which are not supported at
        # the moment. Just verify that LLDB doesn't report an incorrect value
        # for them and just treats them as normal variables (which will lead
        # to linker errors as they are not defined anywhere).
        self.expect("expr A::int128_max", error=True,
                    substrs=["Couldn't lookup symbols:"])
        self.expect("expr A::uint128_max", error=True,
                    substrs=["Couldn't lookup symbols:"])
        self.expect("expr A::int128_min", error=True,
                    substrs=["Couldn't lookup symbols:"])
        self.expect("expr A::uint128_min", error=True,
                    substrs=["Couldn't lookup symbols:"])
