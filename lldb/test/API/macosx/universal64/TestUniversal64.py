from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Universal64TestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def do_test(self):
        exe = self.getBuildArtifact("fat.out")
        target = self.dbg.CreateTarget(exe)

        # Make sure the binary and the dSYM are in the image list.
        self.expect("image list", patterns=["fat.out", "fat.out.dSYM"])

        # The dynamic loader doesn't support fat64 executables so we can't
        # actually launch them here.

    # The Makefile manually invokes clang.
    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test_universal64_executable(self):
        """Test fat64 universal executable"""
        self.build(debug_info="dsym")
        self.do_test()

    # The Makefile manually invokes clang.
    @skipIfAsan
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    def test_universal64_dsym(self):
        """Test fat64 universal dSYM"""
        self.build(debug_info="dsym", dictionary={"FAT64_DSYM": "1"})
        self.do_test()
