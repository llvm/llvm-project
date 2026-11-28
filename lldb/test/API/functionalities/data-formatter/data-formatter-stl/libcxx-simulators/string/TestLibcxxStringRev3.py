from lldbsuite.test.lldbtest import TestBase
import libcxx_string_simulator


class LibcxxStringRev3TestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True


libcxx_string_simulator.add_tests(LibcxxStringRev3TestCase, revision=3)
