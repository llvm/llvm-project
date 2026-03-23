import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os

class TestSwiftFModuleFlags(TestBase):
    @skipIf(macos_version=["<", "14.0"])
    @skipIfDarwinEmbedded
    @swiftTest
    @skipIfWindows
    def test(self):
        """Test that -fmodule flags get stripped out"""
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, 'main')
        self.expect("expression 1", substrs=["1"])

        # Scan through the types log.

        self.filecheck_log(log, __file__)
#       CHECK: main.swift{{.*}} PCM validation is
#       CHECK: main.swift{{.*}} -DMARKER1
#       CHECK-NOT: -fno-implicit-modules
#       CHECK-NOT: -fno-implicit-module-maps
#       CHECK: main.swift{{.*}} -DMARKER2
