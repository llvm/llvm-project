import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxVectorDataFormatterSimulatorTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "return 0", lldb.SBFileSpec("main.cpp"))

        self.expect(
            "frame variable legacy_layout0",
            substrs=["size=0"],
        )
        self.expect(
            "frame variable legacy_layout1",
            substrs=["size=1", "[0] = 10"],
        )
        self.expect(
            "frame variable legacy_layout2",
            substrs=["size=2", "[0] = -10", "[1] = -20"],
        )
        self.expect(
            "frame variable legacy_layout3",
            substrs=["size=3", "[0] = 56", "[1] = 10", "[2] = 87"],
        )

        self.expect(
            "frame variable pointer_based_layout0",
            substrs=["size=0"],
        )
        self.expect(
            "frame variable pointer_based_layout1",
            substrs=["size=1", "[0] = 10"],
        )
        self.expect(
            "frame variable pointer_based_layout2",
            substrs=["size=2", "[0] = -10", "[1] = -20"],
        )
        self.expect(
            "frame variable pointer_based_layout3",
            substrs=["size=3", "[0] = 56", "[1] = 10", "[2] = 87"],
        )

        self.expect(
            "frame variable size_based_layout0",
            substrs=["size=0"],
        )
        self.expect(
            "frame variable size_based_layout1",
            substrs=["size=1", "[0] = 10"],
        )
        self.expect(
            "frame variable size_based_layout2",
            substrs=["size=2", "[0] = -10", "[1] = -20"],
        )
        self.expect(
            "frame variable size_based_layout3",
            substrs=["size=3", "[0] = 56", "[1] = 10", "[2] = 87"],
        )
