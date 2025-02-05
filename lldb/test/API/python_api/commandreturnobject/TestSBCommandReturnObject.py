import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SBCommandReturnObjectTest(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def test(self):
        res = lldb.SBCommandReturnObject()
        self.assertEqual(res.GetCommand(), "")

        ci = self.dbg.GetCommandInterpreter()
        ci.HandleCommand("help help", res)
        self.assertTrue(res.Succeeded())
        self.assertEqual(res.GetCommand(), "help help")
