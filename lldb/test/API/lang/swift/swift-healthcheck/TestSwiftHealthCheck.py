import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftHealthCheck(TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    @skipIfDarwinEmbedded
    def test(self):
        """Test that an underspecified triple is upgraded with a version number.
        """
        self.build()

        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'main')
        self.expect("p 1")
        result = lldb.SBCommandReturnObject()
        ret_val = self.dbg.GetCommandInterpreter().HandleCommand("swift-healthcheck", result)
        log = result.GetOutput()[:-1].split(" ")[-1]
        self.assertEquals(log[-4:], ".log")
        import io, re
        logfile = io.open(log, "r", encoding='utf-8')
        good = 0
        bad = 0
        for line in logfile:
            if re.search('swift-healthcheck', line):
                good += 1
                continue
            if re.search('Unsupported mixing"', line):
                bad += 1
                break
        self.assertGreater(good, 1)
        self.assertEquals(bad, 0)
