import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestOrderedSet(TestBase):
  mydir = TestBase.compute_mydir(__file__)

  def test_ordered_set(self):
    self.build()
    src_file = "main.m"
    src_file_spec = lldb.SBFileSpec(src_file)
    (target, process, thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
          "break here", src_file_spec, exe_name = "a.out")
    frame = thread.GetSelectedFrame()
    self.expect("expr -d run -- orderedSet", substrs=["3 elements"])
    self.expect("expr -d run -- *orderedSet", substrs=["(int)1", "(int)2", "(int)3"])
