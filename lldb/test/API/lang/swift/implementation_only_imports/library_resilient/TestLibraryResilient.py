# TestLibraryResilient.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2019 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test `@_implementationOnly import` in a resilient library used by the main executable
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import time
import unittest2

class TestLibraryResilient(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def launch_info(self):
        info = lldb.SBLaunchInfo([])

        if self.getPlatform() == "freebsd" or self.getPlatform() == "linux":
            # LD_LIBRARY_PATH must be set so the shared libraries are found on
            # startup
            library_path = os.environ.get("LD_LIBRARY_PATH", "")
            if library_path:
                library_path += ":"
            library_path += self.getBuildDir()

            info.SetEnvironmentEntries(["LD_LIBRARY_PATH=" + library_path], True)

        return info

    @swiftTest
    @expectedFailureOS(no_match(["macosx"])) # Requires Remote Mirrors support
    def test_implementation_only_import_library(self):
        """Test `@_implementationOnly import` in a resilient library used by the main executable
        
        See the ReadMe.md in the parent directory for more information.
        """

        self.build()
        def cleanup():
            lldbutil.execute_command("make cleanup")
        self.addTearDownHook(cleanup)
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"), self.launch_info())

        # This test is deliberately checking what the user will see, rather than
        # the structure provided by the Python API, in order to test the recovery.
        self.expect("fr var", substrs=[
            "(SomeLibrary.ContainsTwoInts) container = {\n  wrapped = (first = 2, second = 3)\n  other = 10\n}",
            "(Int) simple = 1"])
        self.expect("e container", substrs=["(SomeLibrary.ContainsTwoInts)", "other = 10"])
        self.expect("e container.wrapped", substrs=["(SomeLibraryCore.TwoInts)", "(first = 2, second = 3)"])

    @swiftTest
    @expectedFailureOS(no_match(["macosx"])) # Requires Remote Mirrors support
    def test_implementation_only_import_library_no_library_module(self):
        """Test `@_implementationOnly import` in a resilient library used by the main executable, after removing the implementation-only library's swiftmodule
        
        See the ReadMe.md in the parent directory for more information.
        """

        self.build()
        os.remove(self.getBuildArtifact("SomeLibraryCore.swiftmodule"))
        os.remove(self.getBuildArtifact("SomeLibraryCore.swiftinterface"))
        def cleanup():
            lldbutil.execute_command("make cleanup")
        self.addTearDownHook(cleanup)
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"), self.launch_info())

        # This test is deliberately checking what the user will see, rather than
        # the structure provided by the Python API, in order to test the recovery.
        self.expect("fr var", substrs=[
            "(SomeLibrary.ContainsTwoInts) container = (other = 10)",
            "(Int) simple = 1"])
        self.expect("e container", substrs=["(SomeLibrary.ContainsTwoInts)", "other = 10"])
        self.expect("e container.wrapped", error=True, substrs=["value of type 'ContainsTwoInts' has no member 'wrapped'"])
