import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LibcxxVectorDataFormatterSimulatorTestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True
    test_cases = {
        "LLDB_TEST_VECTOR_WITHOUT_LAYOUT_DATA_MEMBER": 0,
        "LLDB_TEST_VECTOR_WITH_POINTER_LAYOUT": 1,
        "LLDB_TEST_VECTOR_WITH_SIZE_LAYOUT": 2,
    }

    def _run_test(self, test_case):
        cxxflags_extras = f"-DLLDB_TEST_CASE={test_case}"
        self.build(dictionary=dict(CXXFLAGS_EXTRAS=cxxflags_extras))
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.cpp")
        )

        self.expect(
            "frame variable v0",
            substrs=["size=0"],
        )
        self.expect(
            "frame variable v1",
            substrs=["size=1", "[0] = 10"],
        )
        self.expect(
            "frame variable v2",
            substrs=["size=2", "[0] = -10", "[1] = -20"],
        )
        self.expect(
            "frame variable v3",
            substrs=["size=3", "[0] = 56", "[1] = 10", "[2] = 87"],
        )

    def test_without_layout_member(self):
        self._run_test(self.test_cases["LLDB_TEST_VECTOR_WITHOUT_LAYOUT_DATA_MEMBER"])

    def test_with_pointer_layout(self):
        self._run_test(self.test_cases["LLDB_TEST_VECTOR_WITH_POINTER_LAYOUT"])

    def test_with_size_layout(self):
        self._run_test(self.test_cases["LLDB_TEST_VECTOR_WITH_SIZE_LAYOUT"])
