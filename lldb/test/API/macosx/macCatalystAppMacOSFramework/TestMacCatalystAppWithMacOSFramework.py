import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestMacCatalystAppWithMacOSFramework(TestBase):
    @skipIf(macos_version=["<", "10.15"])
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    # There is a Clang driver change missing on llvm.org.
    @expectedFailureAll(bugnumber="rdar://problem/54986190>")
    def test(self):
        """Test the x86_64-apple-ios-macabi target linked against a macos dylib"""
        self.build()
        log = self.getBuildArtifact("packets.log")
        self.expect("log enable gdb-remote packets -f " + log)
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.c"))
        arch = self.getArchitecture()
        self.expect(
            "image list -t -b",
            patterns=[
                arch + r".*-apple-ios.*-macabi a\.out",
                arch + r".*-apple-macosx.* libfoo.dylib[^(]",
            ],
        )
        self.expect("fr v s", "Hello macCatalyst")
        self.expect("expression s", "Hello macCatalyst")
        self.check_debugserver(log)

    def check_debugserver(self, log):
        """scan the debugserver packet log"""
        process_info = lldbutil.packetlog_get_process_info(log)
        self.assertIn("ostype", process_info)
        self.assertEquals(process_info["ostype"], "maccatalyst")

        aout_info = None
        libfoo_info = None
        dylib_info = lldbutil.packetlog_get_dylib_info(log)
        for image in dylib_info["images"]:
            if image["pathname"].endswith("a.out"):
                aout_info = image
            if image["pathname"].endswith("libfoo.dylib"):
                libfoo_info = image
        self.assertTrue(aout_info)
        self.assertTrue(libfoo_info)
        self.assertEquals(aout_info["min_version_os_name"], "maccatalyst")
        self.assertEquals(libfoo_info["min_version_os_name"], "macosx")
