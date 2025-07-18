import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


@skipUnlessDarwin
@skipIf(bugnumber='rdar://156138054')  # fails sometimes in swift PR testing
class TestSwiftFoundationTypeNotification(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    @swiftTest
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
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
        name = self.frame().FindVariable("name")
        # This is a "don't crash" test.
        child = name.GetChildAtIndex(0)
        self.assertEquals(name.GetSummary(), '"MyNotification"')
