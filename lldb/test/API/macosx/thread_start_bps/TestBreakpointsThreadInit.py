"""Test that we get thread names when interrupting a process."""

import time
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestInterruptThreadNames(TestBase):
    @skipUnlessDarwin
    def test_internal_bps_resolved(self):
        self.build()

        source_file = lldb.SBFileSpec("main.c")
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "initial hello", source_file
        )

        thread_start_func_names = [
            "start_wqthread",
            "_pthread_wqthread",
            "_pthread_start",
        ]
        known_module_names = [
            "libsystem_c.dylib",
            "libSystem.B.dylib",
            "libsystem_pthread.dylib",
        ]
        bps = []
        for func in thread_start_func_names:
            for module in known_module_names:
                bps.append(target.BreakpointCreateByName(func, module))
        num_resolved = 0
        for bp in bps:
            num_resolved += bp.GetNumResolvedLocations()
        self.assertGreater(num_resolved, 0)
