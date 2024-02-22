import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftPrivateGenericType(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    @skipUnlessDarwin
    @swiftTest
    def test_private_generic_type(self):
        """ Test that evaluating an expression without binding generic 
        parameters works for private generic types"""
        invisible_swift = self.getBuildArtifact("Private.swift")
        import shutil
        shutil.copyfile("Private.swift", invisible_swift)
        self.build()
        os.unlink(invisible_swift)
        os.unlink(self.getBuildArtifact("Private.swiftmodule"))
        os.unlink(self.getBuildArtifact("Private.swiftinterface"))

        target, process, _, _ = lldbutil.run_to_source_breakpoint(self, 
                'break here for struct', lldb.SBFileSpec('Public.swift'),
                extra_images=['Public'])
        # Make sure this fails without generic expression evaluation.
        self.expect("e --bind-generic-types true -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
        # Test that not binding works.
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Public.StructWrapper<T>", 
                             'name = "The invisible struct."'])
        # Test that the "auto" behavior also works.
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.StructWrapper<T>", 
                             'name = "The invisible struct."'])
        # Test that the default (should be the auto option) also works.
        self.expect("e -- self", substrs=["Public.StructWrapper<T>", 
                                          'name = "The invisible struct."'])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for class', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types true -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Public.ClassWrapper<Private.InvisibleStruct>", 
                             'name = "The invisible struct."'])
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.ClassWrapper<Private.InvisibleStruct>", 
                             'name = "The invisible struct."'])
        self.expect("e -- self", 
                    substrs=["Public.ClassWrapper<Private.InvisibleStruct>", 
                             'name = "The invisible struct."'])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for two generic parameters', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types true -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Public.TwoGenericParameters",
                             "<Private.InvisibleClass, Private.InvisibleStruct>", 
                             'name = "The invisible class."',
                             "someNumber = 42"])
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.TwoGenericParameters",
                             "<Private.InvisibleClass, Private.InvisibleStruct>", 
                             'name = "The invisible class."',
                             "someNumber = 42"])
        self.expect("e -- self", 
                    substrs=["Public.TwoGenericParameters",
                             "<Private.InvisibleClass, Private.InvisibleStruct>", 
                             'name = "The invisible class."',
                             "someNumber = 42"])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for three generic parameters', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types true -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Public.ThreeGenericParameters",
                             "<T, U, V>", 
                             'name = "The invisible class."',
                             "someNumber = 42",
                             'name = "The invisible struct."',
                             "v = true"])
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.ThreeGenericParameters",
                             "<T, U, V>", 
                             'name = "The invisible class."',
                             "someNumber = 42",
                             'name = "The invisible struct."',
                             "v = true"])
        self.expect("e -- self", 
                    substrs=["Public.ThreeGenericParameters",
                             "<T, U, V>", 
                             'name = "The invisible class."',
                             "someNumber = 42",
                             'name = "The invisible struct."',
                             "v = true"])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for four generic parameters', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types true -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Public.FourGenericParameters",
                             "<T, U, V, W>", 
                             'name = "The invisible struct."',
                             'name = "The invisible class."',
                             "someNumber = 42",
                             "v = 3 values", "One", "two", "three",
                             "w = 482"])
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.FourGenericParameters",
                             "<T, U, V, W>", 
                             'name = "The invisible struct."',
                             'name = "The invisible class."',
                             "someNumber = 42",
                             "v = 3 values", "One", "two", "three",
                             "w = 482"])
        self.expect("e -- self", 
                    substrs=["Public.FourGenericParameters",
                             "<T, U, V, W>", 
                             'name = "The invisible struct."',
                             'name = "The invisible class."',
                             "someNumber = 42",
                             "v = 3 values", "One", "two", "three",
                             "w = 482"])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for non-generic', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Could not evaluate the expression without binding generic types."], 
                    error=True)

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for nested generic parameters', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Could not evaluate the expression without binding generic types."], 
                    error=True)

        # Check that if both binding and not binding the generic type parameters fail, we report 
        # the "bind generic params" error message, as that's the default case that runs first.
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
