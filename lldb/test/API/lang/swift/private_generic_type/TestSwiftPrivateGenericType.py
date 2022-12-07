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
                             "The invisible man."])
        # Test that the "auto" behavior also works.
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.StructWrapper<T>", 
                             "The invisible man."])
        # Test that the default (should be the auto option) also works.
        self.expect("e -- self", substrs=["Public.StructWrapper<T>", 
                                          "The invisible man."])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for class', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types true -- self", 
                    substrs=["Couldn't realize Swift AST type of self."], 
                    error=True)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Public.ClassWrapper<Private.InvisibleStruct>", 
                             "The invisible man."])
        self.expect("e --bind-generic-types auto -- self", 
                    substrs=["Public.ClassWrapper<Private.InvisibleStruct>", 
                             "The invisible man."])
        self.expect("e -- self", 
                    substrs=["Public.ClassWrapper<Private.InvisibleStruct>", 
                             "The invisible man."])

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for non-generic', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Could not evaluate the expression without binding generic types."], 
                    error=True)

        breakpoint = target.BreakpointCreateBySourceRegex(
            'break here for two generic parameters', lldb.SBFileSpec('Public.swift'), None)
        lldbutil.continue_to_breakpoint(process, breakpoint)
        self.expect("e --bind-generic-types false -- self", 
                    substrs=["Could not evaluate the expression without binding generic types."], 
                    error=True)

