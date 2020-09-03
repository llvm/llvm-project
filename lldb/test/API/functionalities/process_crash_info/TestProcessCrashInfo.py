"""
Test lldb process crash info.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
from lldbsuite.test import lldbtest


class PlatformProcessCrashInfoTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        TestBase.setUp(self)
        self.runCmd("settings set auto-confirm true")
        self.source = "main.c"
        self.line = line_number(self.source, '// break here')

    def tearDown(self):
        self.runCmd("settings clear auto-confirm")
        TestBase.tearDown(self)

    @skipIfAsan # The test process intentionally double-frees.
    @skipUnlessDarwin
    def test_cli(self):
        """Test that `process status --verbose` fetches the extended crash
        information dictionary from the command-line properly."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.expect("file " + exe,
                    patterns=["Current executable set to .*a.out"])

        self.expect('process launch',
                    patterns=["Process .* launched: .*a.out"])

        self.expect('process status --verbose',
                    patterns=["\"message\".*pointer being freed was not allocated"])


    @skipIfAsan # The test process intentionally hits a memory bug.
    @skipUnlessDarwin
    def test_api(self):
        """Test that lldb can fetch a crashed process' extended crash information
        dictionary from the api properly."""
        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        target.LaunchSimple(None, None, os.getcwd())

        stream = lldb.SBStream()
        self.assertTrue(stream)

        process = target.GetProcess()
        self.assertTrue(process)

        crash_info = process.GetExtendedCrashInformation()

        error = crash_info.GetAsJSON(stream)

        self.assertTrue(error.Success())

        self.assertTrue(crash_info.IsValid())

        self.assertIn("pointer being freed was not allocated", stream.GetData())

    # dyld leaves permanent crash_info records when testing on device.
    @skipIfDarwinEmbedded
    def test_on_sane_process(self):
        """Test that lldb doesn't fetch the extended crash information
        dictionary from a 'sane' stopped process."""
        self.build()
        target, _, _, _ = lldbutil.run_to_line_breakpoint(self, lldb.SBFileSpec(self.source),
                                        self.line)

        stream = lldb.SBStream()
        self.assertTrue(stream)

        process = target.GetProcess()
        self.assertTrue(process)

        crash_info = process.GetExtendedCrashInformation()

        error = crash_info.GetAsJSON(stream)
        self.assertFalse(error.Success())
        self.assertIn("No structured data.", error.GetCString())
