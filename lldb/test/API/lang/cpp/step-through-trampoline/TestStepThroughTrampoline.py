import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil

class StepThroughTrampoline(TestBase):
    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                   "// Set a breakpoint here",
                                   lldb.SBFileSpec("main.cpp"),
                                   extra_images=["foo"])
        thread.StepInto()

        foo_line = line_number("foo.cpp", '// End up here')
        self.expect("frame info", substrs=["foo.cpp:{}:".format(foo_line)])
