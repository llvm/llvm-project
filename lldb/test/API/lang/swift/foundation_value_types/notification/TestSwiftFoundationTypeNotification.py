import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


@skipUnlessDarwin
class TestSwiftFoundationTypeNotification(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect("log enable lldb types -v")
        # global
        self.expect("target variable -d run g_notification",
                    substrs=['name = ', '"MyNotification"',
                             'object = nil',
                             'userInfo = 0 key/value pairs'])
        self.expect("expression -d run -- g_notification",
                    substrs=['name = ', '"MyNotification"',
                             'object = nil',
                             'userInfo = 0 key/value pairs'])
        # local
        self.expect("frame variable -d run -- notification",
                    substrs=['name = ', '"MyNotification"',
                             'object = nil',
                             'userInfo = 0 key/value pairs'])
        self.expect("expression -d run -- notification",
                    substrs=['name = ', '"MyNotification"',
                             'object = nil',
                             'userInfo = 0 key/value pairs'])
