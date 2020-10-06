import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestMacCatalyst(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipIf(macos_version=["<", "10.15"])
    @skipUnlessDarwin
    @skipIfDarwinEmbedded
    @skipIfReproducer # This is hitting https://bugs.python.org/issue22393
    def test_macabi(self):
        """Test the x86_64-apple-ios-macabi target linked against a macos dylib"""
        self.build()
        log = self.getBuildArtifact('packets.log')
        self.expect("log enable gdb-remote packets -f "+log)
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.c'))
        self.expect("image list -t -b",
                    patterns=[self.getArchitecture() +
                              r'.*-apple-ios.*-macabi a\.out'])
        self.expect("fr v s", substrs=["Hello macCatalyst"])
        self.expect("p s", substrs=["Hello macCatalyst"])
        self.check_debugserver(log)

    def check_debugserver(self, log):
        """scan the debugserver packet log"""
        process_info = lldbutil.packetlog_get_process_info(log)
        self.assertTrue('ostype' in process_info)
        self.assertEquals(process_info['ostype'], 'maccatalyst')

        aout_info = None
        dylib_info = lldbutil.packetlog_get_dylib_info(log)
        for image in dylib_info['images']:
            if image['pathname'].endswith('a.out'):
                aout_info = image
        self.assertTrue(aout_info)
        self.assertEquals(aout_info['min_version_os_name'], 'maccatalyst')
