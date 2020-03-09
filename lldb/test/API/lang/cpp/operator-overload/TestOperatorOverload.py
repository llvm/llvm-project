import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestOperatorOverload(TestBase):
  mydir = TestBase.compute_mydir(__file__)

  def test_overload(self):
    self.build()
    (target, process, thread,
      main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
        "break here", lldb.SBFileSpec("b.cpp"))
    frame = thread.GetSelectedFrame()
    value = frame.EvaluateExpression("x == nil")
    self.assertTrue(str(value.GetError())
      .find("comparison between NULL and non-pointer ('Tinky' and NULL)")
        != -1)
    self.assertTrue(str(value.GetError())
      .find("invalid operands to binary expression ('Tinky' and")
        != -1)
    self.assertFalse(value.GetError().Success())
