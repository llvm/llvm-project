import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftVariadicGenerics(TestBase):

    @skipUnlessDarwin
    @swiftTest
    def test(self):
        self.build()

        target,  process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('a.swift'))

        # f1(args: a, b)
        self.expect("frame variable",
                    substrs=["Pack{(a.A, a.B)}", "args", "i = 23", "d = 2.71"])

        # f2(us: a, vs: b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A)}", "us", "i = 23",
                             "Pack{(a.B)}", "vs", "d = 2.71"])

        # f3(ts: a, b, more_ts: a, b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A, a.B)}", "ts", "i = 23", # FIXME! "d = 2.71",
                             "Pack{(a.A, a.B)}", "more_ts", "i = 23", "d = 2.71"])

        # f4(uvs: (a, b), (a, b))
        #process.Continue()
        #self.expect("frame variable",
        #            substrs=[])

        # f5(ts: (a, b), (42, b))
        process.Continue()
        self.expect("frame variable",
                    substrs=[# FIXME: "Pack{(a.A, a.B), (Int, a,B)}",
                             "ts", "i = 23", "d = 2.71"
                             # FIXME: "42", "d = 2.71"
                    ])

        # f6(us: a, more_us: a, vs: b, b)
        #process.Continue()
        #self.expect("frame variable",
        #            substrs=[])

        # f7(us: a, vs: 1, b, more_us: a, more_vs: 2, b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A)}", "us", # FIXME: "i = 23"
                             "Pack{(Int, a.B)}", "vs", "= 1", "d = 2.71",
                             "Pack{(a.A)}", "more_us", #FIXME: "i = 23"
                             "Pack{(Int, a.B)}", "more_vs", "= 2", "d = 2.71"
                    ])
                        
        
