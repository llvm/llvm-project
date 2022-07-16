import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftTaggedPointer(lldbtest.TestBase):

    @swiftTest
    # This test depends on NSObject, so it is not available on non-Darwin
    # platforms.
    @skipUnlessDarwin
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        self.expect('frame variable -d run -- a', substrs=['Int64(3)'])
        self.expect('frame variable -d run -- b', substrs=['Int64(3)'])
         
        self.expect('frame variable -d run -- c', substrs=['"hi"'])
        self.expect('frame variable -d run -- d', substrs=['"hi"'])
         
        self.expect('expr -d run -- a', substrs=['Int64(3)'])
        self.expect('expr -d run -- b', substrs=['Int64(3)'])
         
        self.expect('expr -d run -- c', substrs=['"hi"'])
        self.expect('expr -d run -- d', substrs=['"hi"'])
