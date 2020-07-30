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
    self.assertFalse(value.GetError().Success())
    self.assertIn("comparison between NULL and non-pointer ('Tinky' and NULL)", str(value.GetError()))
    self.assertIn("invalid operands to binary expression ('Tinky' and", str(value.GetError()))
