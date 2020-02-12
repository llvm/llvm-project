"""
Use lldb Python API to test base class resolution for ObjC classes
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ObjCDynamicValueTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)

        self.line = line_number('main.m', '// Set breakpoint here.')

    @skipUnlessDarwin
    @add_test_categories(['pyapi'])
    def test_get_baseclass(self):
        """Test fetching ObjC dynamic values."""
        if self.getArchitecture() == 'i386':
            # rdar://problem/9946499
            self.skipTest("Dynamic types for ObjC V1 runtime not implemented")

        self.build()
        exe = self.getBuildArtifact("a.out")

        # Create a target from the debugger.

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Set up our breakpoints:

        target.BreakpointCreateByLocation('main.m', self.line)
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        var = self.frame().FindVariable("foo")
        var_ptr_type = var.GetType()
        var_pte_type = var_ptr_type.GetPointeeType()
        self.assertTrue(
            var_ptr_type.GetNumberOfDirectBaseClasses() == 1,
            "Foo * has one base class")
        self.assertTrue(
            var_pte_type.GetNumberOfDirectBaseClasses() == 1,
            "Foo has one base class")

        self.assertTrue(var_ptr_type.GetDirectBaseClassAtIndex(
            0).IsValid(), "Foo * has a valid base class")
        self.assertTrue(var_pte_type.GetDirectBaseClassAtIndex(
            0).IsValid(), "Foo * has a valid base class")

        self.assertTrue(var_ptr_type.GetDirectBaseClassAtIndex(0).GetName() == var_pte_type.GetDirectBaseClassAtIndex(
            0).GetName(), "Foo and its pointer type don't agree on their base class")
