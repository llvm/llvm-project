"""
Test that LLDB can launch a linux executable through the dynamic loader where
the main executable has an extra exported "_r_debug" symbol that used to mess
up shared library loading with DYLDRendezvous and the POSIX dynamic loader
plug-in. What used to happen is that any shared libraries other than the main
executable and the dynamic loader and VSDO would not get loaded. This test
checks to make sure that we still load libraries correctly when we have
multiple "_r_debug" symbols. See comments in the main.cpp source file for full
details on what the problem is.
"""

import lldb
import os

from lldbsuite.test import lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestDyldWithMultipleRDebug(TestBase):
    @skipIf(oslist=no_match(["linux"]))
    @no_debug_info_test
    def test(self):
        self.build()
        # Run to a breakpoint in main.cpp to ensure we can hit breakpoints
        # in the main executable. Setting breakpoints by file and line ensures
        # that the main executable was loaded correctly by the dynamic loader
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Break here", lldb.SBFileSpec("main.cpp"),
            extra_images=["testlib"]
        )
        # Set breakpoints both on shared library function to ensure that
        # we hit a source breakpoint in the shared library which only will
        # happen if we load the shared library correctly in the dynamic
        # loader.
        lldbutil.continue_to_source_breakpoint(
            self, process, "// Library break here",
            lldb.SBFileSpec("library_file.cpp", False)
        )
