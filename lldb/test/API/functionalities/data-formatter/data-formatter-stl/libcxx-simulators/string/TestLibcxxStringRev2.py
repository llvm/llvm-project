from lldbsuite.test.lldbtest import TestBase
import libcxx_string_simulator


class LibcxxStringRev2TestCase(TestBase):
    SHARED_BUILD_TESTCASE = False
    NO_DEBUG_INFO_TESTCASE = True


libcxx_string_simulator.add_tests(LibcxxStringRev2TestCase, revision=2)
