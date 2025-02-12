import os
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestSwiftOriginallyDefinedIn(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test that types with the @_originallyDefinedIn attribute can still be found in metadata"""

        self.build()
        self.runCmd("setting set symbols.swift-enable-ast-context false")
        filespec = lldb.SBFileSpec("main.swift")
        target, process, thread, breakpoint1 = lldbutil.run_to_source_breakpoint(
            self, "break here", filespec
        )
        self.expect("frame variable a", substrs=["a = (i = 10)"])
        self.expect("frame variable b", substrs=["b = (i = 20)"])
        self.expect("frame variable d", substrs=["d = (i = 30)"])
        self.expect("frame variable e", substrs=["i = 50"])
        self.expect("frame variable f", substrs=["i = 40"])
        self.expect("frame variable g", substrs=["i = 60"])
        self.expect("frame variable h", substrs=["t = (i = 50)", "u = (i = 70)"])
        self.expect("frame variable i", substrs=["(i = 10)", "(i = 40)", "(i = 50)"])
        self.expect(
            "frame variable complex",
            substrs=[
                "t = t {",
                "t = {",
                "t = (i = 70)",
                "u = (i = 30)",
                "u = t {",
                "t = (i = 50)",
            ],
        )
    
    @swiftTest
    def test_expr(self):
        """Test that types with the @_originallyDefinedIn attribute can still be found in metadata"""
    
        self.build()
        filespec = lldb.SBFileSpec("main.swift")
        target, process, thread, breakpoint1 = lldbutil.run_to_source_breakpoint(
            self, "break here", filespec
        )
        self.expect("expr a", substrs=["(i = 10)"])
        self.expect("expr b", substrs=["(i = 20)"])
        self.expect("expr d", substrs=["(i = 30)"])
        self.expect("expr e", substrs=["(i = 50)"])
        self.expect("expr f", substrs=["i = 40"])
        self.expect("expr g", substrs=["i = 60"])
        self.expect("expr i", substrs=["(i = 10)", "(i = 40)", "(i = 50)"])
        self.expect(
            "expr complex",
            substrs=[
                "t = t {",
                "t = {",
                "t = (i = 70)",
                "u = (i = 30)",
                "u = t {",
                "t = (i = 50)",
            ],
        )
    
    @swiftTest
    def test_expr_from_generic(self):
         """Test that types with the @_originallyDefinedIn attribute can still be found in metadata"""

         self.build()
         filespec = lldb.SBFileSpec("main.swift")
         target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
             self, "break for generic", filespec
         )
         self.expect("expr t", substrs=["(i = 10)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["(i = 20)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["(i = 30)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["(i = 50)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["(i = 40)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["(i = 60)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["t = (i = 50)", "u = (i = 70)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect("expr t", substrs=["(i = 10)", "(i = 40)", "(i = 50)"])
         lldbutil.continue_to_breakpoint(process, bkpt)
         self.expect(
             "expr t",
             substrs=[
                "t = t {",
                "t = {",
                "t = (i = 70)",
                "u = (i = 30)",
                "u = t {",
                "t = (i = 50)",
             ],
         )
