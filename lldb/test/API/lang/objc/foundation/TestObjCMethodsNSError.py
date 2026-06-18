"""
Test more expression command sequences with objective-c.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class FoundationTestCaseNSError(TestBase):
    @expectedFailureAll(archs=["i[3-6]86"], bugnumber="<rdar://problem/28814052>")
    def test_runtime_types(self):
        """Test commands that require runtime types"""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// Break here for NSString tests", lldb.SBFileSpec("main.m", False)
        )

        # Test_NSString:
        self.runCmd("thread backtrace")
        self.expect("expression [str length]", patterns=[r"\(NSUInteger\) \$.* ="])
        self.expect("expression str.length")
        self.expect('expression str = [NSString stringWithCString: "new"]')
        self.expect(
            'po [NSError errorWithDomain:@"Hello" code:35 userInfo:@{@"NSDescription" : @"be completed."}]',
            substrs=["Error Domain=Hello", "Code=35", "be completed."],
        )
        self.runCmd("process continue")

    @expectedFailureAll(archs=["i[3-6]86"], bugnumber="<rdar://problem/28814052>")
    def test_NSError_p(self):
        """Test that p of the result of an unknown method does require a cast."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line", lldb.SBFileSpec("main.m", False)
        )
        self.expect(
            "expression [NSError thisMethodIsntImplemented:0]",
            error=True,
            patterns=[
                "no known method",
                "cast the message send to the method's return type",
            ],
        )
        self.runCmd("process continue")

    @skipIfOutOfTreeDebugserver
    def test_runtime_types_efficient_memreads(self):
        # Test that we use an efficient reading of memory when reading
        # Objective-C method descriptions.
        logfile = os.path.join(self.getBuildDir(), "log.txt")
        self.runCmd(f"log enable -f {logfile} gdb-remote packets process")
        self.addTearDownHook(lambda: self.runCmd("log disable gdb-remote packets"))

        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// Break here for NSString tests", lldb.SBFileSpec("main.m", False)
        )

        self.runCmd(f"proc plugin packet send StartTesting", check=False)
        self.expect('expression str = [NSString stringWithCString: "new"]')
        self.runCmd(f"proc plugin packet send EndTesting", check=False)

        self.assertTrue(os.path.exists(logfile))
        log_text = open(logfile).read()
        log_text = log_text.split("StartTesting", 1)[-1].split("EndTesting", 1)[0]

        # This test is only checking that the packet it used at all (and that
        # no errors are produced). It doesn't check that the packet is being
        # used to solve a problem in an optimal way.
        self.assertIn("MultiMemRead:", log_text)
        self.assertNotIn("MultiMemRead error", log_text)
