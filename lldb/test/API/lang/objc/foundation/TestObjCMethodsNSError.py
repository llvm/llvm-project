"""
Test more expression command sequences with objective-c.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class FoundationTestCaseNSError(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(archs=["i[3-6]86"], bugnumber="<rdar://problem/28814052>")
    def test_runtime_types(self):
        """Test commands that require runtime types"""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
                self, '// Break here for NSString tests',
                lldb.SBFileSpec('main.m', False))

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.expect("expression [str length]",
                    patterns=["\(NSUInteger\) \$.* ="])
        self.expect("expression str.length")
        self.expect('expression str = [NSString stringWithCString: "new"]')
        self.expect(
            'po [NSError errorWithDomain:@"Hello" code:35 userInfo:@{@"NSDescription" : @"be completed."}]',
            substrs=[
                "Error Domain=Hello",
                "Code=35",
                "be completed."])
        self.runCmd("process continue")

    @expectedFailureAll(archs=["i[3-6]86"], bugnumber="<rdar://problem/28814052>")
    def test_NSError_p(self):
        """Test that p of the result of an unknown method does require a cast."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
                self, '// Set break point at this line',
                lldb.SBFileSpec('main.m', False))
        self.expect("p [NSError thisMethodIsntImplemented:0]", error=True, patterns=[
                    "no known method", "cast the message send to the method's return type"])
        self.runCmd("process continue")
