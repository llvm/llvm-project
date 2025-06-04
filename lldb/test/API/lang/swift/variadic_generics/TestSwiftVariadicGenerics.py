import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftVariadicGenerics(TestBase):

    NO_DEBUG_INFO_TESTCASE = True
    @skipUnlessDarwin
    @swiftTest
    @skipIfAsan # rdar://152465885 Address Sanitizer assert doing `expr --bind-generic-types=false -- 0`
    def test(self):
        self.build()

        target,  process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('a.swift'))

        # f1(args: a, b)
        self.expect("frame variable",
                    substrs=["Pack{(a.A, a.B)}", "args", "i = 23", "d = 2.71"])

        # Test that an expression can be set up in a variadic environment.
        self.expect("expr --bind-generic-types=true -- 0", substrs=["0"])
        self.expect("expr --bind-generic-types=false -- 0", substrs=["0"])
        # FIXME: crashes the compiler.
        #self.expect("expr --bind-generic-params=false -- (repeat each args)",
        #            substrs=["args"])

        # f2(us: a, vs: b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A)}", "us", "i = 23",
                             "Pack{(a.B)}", "vs", "d = 2.71"])

        # f3(ts: a, b, more_ts: a, b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A, a.B)}", "ts", "i = 23", "d = 2.71",
                             "Pack{(a.A, a.B)}", "more_ts", "i = 23", "d = 2.71"])

        # f4(uvs: (a, b), (a, b))
        process.Continue()
        self.expect("frame variable",
                    substrs=[""])

        # f5(ts: (a, b), (42, b))
        process.Continue()
        self.expect("frame variable",
                    substrs=[# FIXME: "Pack{(a.A, a.B), (Int, a,B)}",
                             "ts", "i = 23", "d = 2.71"
                             # FIXME: "42", "d = 2.71"
                    ])

        # f6(us: a, more_us: a, vs: b, b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A)}", "us", "i = 23",
                             "Pack{(a.A)}", "more_us", "i = 23",
                             "Pack{(a.B, a.B)}", "vs", "d = 2.71", "d = 2.71"
                             ])

        # f7(us: a, vs: 1, b, more_us: a, more_vs: 2, b)
        process.Continue()
        self.expect("frame variable",
                    substrs=["Pack{(a.A)}", "us", "i = 23",
                             "Pack{(Int, a.B)}", "vs", "= 1", "d = 2.71",
                             "Pack{(a.A)}", "more_us", "i = 23",
                             "Pack{(Int, a.B)}", "more_vs", "= 2", "d = 2.71"
                    ])

        # f8(<specialized self>)
        process.Continue()
        # specialized global
        process.Continue()
        self.expect("target variable s",
                    substrs=["vals", "0 = 23", "1 = 2.71"])

        # f9(s: S<repeat each T>)
        process.Continue()
        self.expect("frame variable", substrs=["t", "0 = 23", "1 = 2.71"])

        # f10<each T>(args: repeat each T)
        process.Continue()
        self.expect(
            "frame variable", substrs=[
                "Pack{(a.A, a.B)}",
                "args",
                "i = 23",
                "d ="
                # FIXME: The wrong value for d is currently shown.
                #"d = 2.71"
            ]
        )
