from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Universal64TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def do_test(self):
        # Get the executable.
        exe = self.getBuildArtifact("fat.out")

        # Create a target.
        self.target = self.dbg.CreateTarget(exe)

        # Create a breakpoint on main.
        main_bp = self.target.BreakpointCreateByName("main")
        self.assertTrue(main_bp, VALID_BREAKPOINT)

        # Make sure the binary and the dSYM are in the image list.
        self.expect("image list ", patterns=['fat.out', 'fat.out.dSYM'])

        # The dynamic loader doesn't support fat64 executables so we can't
        # actually launch them here.

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipIf(macos_version=["<", "11.0"])
    def test_universal64_executable(self):
        """Test fat64 universal executable"""
        self.build(debug_info="dsym")
        self.do_test()

    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipIf(macos_version=["<", "11.0"])
    def test_universal64_dsym(self):
        """Test fat64 universal dSYM"""
        self.build(debug_info="dsym", dictionary={'FAT64_DSYM': '1'})
        self.do_test()
