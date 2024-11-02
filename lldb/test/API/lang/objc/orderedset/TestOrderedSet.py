import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class TestOrderedSet(TestBase):

  def test_ordered_set(self):
    self.build()
    self.run_tests()

  @skipUnlessDarwin
  def test_ordered_set_no_const(self):
    disable_constant_classes = {
        'CC':
        'xcrun clang',  # FIXME: Remove when flags are available upstream.
        'CFLAGS_EXTRAS':
        '-fno-constant-nsnumber-literals ' +
        '-fno-constant-nsarray-literals ' +
        '-fno-constant-nsdictionary-literals'
    }
    self.build(dictionary=disable_constant_classes)
    self.run_tests()

  def run_tests(self):
    src_file = "main.m"
    src_file_spec = lldb.SBFileSpec(src_file)
    (target, process, thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(self,
          "break here", src_file_spec, exe_name = "a.out")
    frame = thread.GetSelectedFrame()
    self.expect("expr -d run -- orderedSet", substrs=["3 elements"])
    self.expect("expr -d run -- *orderedSet", substrs=["(int)1", "(int)2", "(int)3"])
