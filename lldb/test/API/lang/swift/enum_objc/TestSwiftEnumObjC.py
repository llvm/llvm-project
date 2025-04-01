import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftDWARFImporterC(lldbtest.TestBase):

    @swiftTest
    @expectedFailureAll(oslist=["linux"], bugnumber="rdar://83444822")
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect('frame var -d run-target -- ascending', substrs=['OrderedAscending'])
        self.expect('frame var -d run-target -- descending', substrs=['OrderedDescending'])
        self.expect('frame var -d run-target -- same', substrs=['OrderedSame'])
        self.expect('expr -d run-target -- ascending', substrs=['OrderedAscending'])
        self.expect('expr -d run-target -- descending', substrs=['OrderedDescending'])
        self.expect('expr -d run-target -- same', substrs=['OrderedSame'])
