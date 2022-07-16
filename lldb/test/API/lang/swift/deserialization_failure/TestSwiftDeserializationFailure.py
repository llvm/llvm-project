import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2


class TestSwiftDeserializationFailure(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    def prepare(self):
        import shutil
        copied_source = self.getBuildArtifact("main.swift")
        shutil.copyfile(os.path.join("Inputs", "main.swift"), copied_source)
        self.build()
        os.unlink(copied_source)
        os.unlink(self.getBuildArtifact("a.swiftmodule"))

    def run_tests(self, target, process):
        static_bkpt = target.BreakpointCreateByName('staticTypes')
        dynamic_bkpt = target.BreakpointCreateByName('dynamicTypes')
        generic_bkpt = target.BreakpointCreateByName('genericTypes')
        lldbutil.continue_to_breakpoint(process, static_bkpt)
        self.expect("fr var i", substrs=["23"])
        self.expect("fr var s", substrs=["(String)", "world"])

        # We should not be able to resolve the types defined in the module.
        lldbutil.continue_to_breakpoint(process, dynamic_bkpt)
        # FIXME: Resurface this error!
        self.expect("fr var c", substrs=[""]) #"<could not resolve type>"])

        lldbutil.continue_to_breakpoint(process, generic_bkpt)
        # FIXME: this is formatted incorrectly.
        self.expect("fr var t", substrs=["(T)"]) #, "world"])

    @swiftTest
    @skipIf(oslist=['windows'])
    @skipIf(debug_info=no_match(["dwarf"]))
    @expectedFailureAll(archs=["arm64", "arm64e", 'arm64_32'], bugnumber="<rdar://problem/58096919>")
    @expectedFailureAll(archs=["arm64", "arm64e", 'arm64_32'], bugnumber="<rdar://problem/58097436>")
    def test_missing_module(self):
        """Test what happens when a .swiftmodule can't be loaded"""
        self.prepare()
        target, process, _, _ = lldbutil.run_to_name_breakpoint(self, 'main')
        self.run_tests(target, process)

    @swiftTest
    @skipIf(oslist=['windows'])
    @skipIf(debug_info=no_match(["dwarf"]))
    @expectedFailureAll(archs=["arm64", "arm64e", 'arm64_32'], bugnumber="<rdar://problem/58096919>")
    @expectedFailureAll(archs=["arm64", "arm64e", 'arm64_32'], bugnumber="<rdar://problem/58097436>")
    def test_damaged_module(self):
        """Test what happens when a .swiftmodule can't be loaded"""
        self.prepare()
        with open(self.getBuildArtifact("a.swiftmodule"), 'w') as mod:
            mod.write('I am damaged.\n')

        target, process, _, _ = lldbutil.run_to_name_breakpoint(self, 'main')
        self.run_tests(target, process)
