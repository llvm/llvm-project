import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil


class TestArchetypeInExpression(TestBase):

    @swiftTest
    def test(self):
        """Tests that a user can refer to the archetypes in their expressions"""         
        self.build()

        # Check that you can refer to all types and function's archetypes.
        _, process, _, breakpoint = lldbutil.run_to_source_breakpoint(self, 
                "break here", lldb.SBFileSpec("main.swift"))
        self.expect("expr First.self", substrs=["Int.Type"])
        self.expect("expr Second.self", substrs=["Double.Type"])
        self.expect("expr Third.self", substrs=["Bool.Type"])
        self.expect("expr T.self", substrs=["String.Type"])
        self.expect("expr U.self", substrs=["[Int].Type"])
        # Assert that frame variable doesn't work
        self.expect("v First.self", 
                substrs=["use of undeclared identifier 'First'"],
                error=True)
        self.expect("v T.self", 
                substrs=["use of undeclared identifier 'T'"],
                error=True)

        # Check that referring to a shadowed archetype works correctly.
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr T.self", substrs=["String.Type"])
        # Assert that frame variable doesn't work
        self.expect("v T.self", 
                substrs=["use of undeclared identifier 'T'"],
                error=True)

        # Check that you refer to archetypes in nested generic functions.
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr T.self", substrs=["Bool.Type"])
        self.expect("expr U.self", substrs=["Double.Type"])

        # Check that you refer to archetypes in nested generic functions with shadowed archetypes.
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr T.self", substrs=["String.Type"])
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("expr T.self", substrs=["Int.Type"])
        
