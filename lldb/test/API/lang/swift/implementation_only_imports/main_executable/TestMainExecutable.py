# TestMainExecutable.py
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
Test `@_implementationOnly import` in the main executable
"""
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import time
import unittest2

@skipIfDarwin # rdar://problem/54322424 Sometimes failing, sometimes truncated output.
class TestMainExecutable(TestBase):
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

    @skipIf(bugnumber="rdar://problem/54322424", # This test is unreliable.
            setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    def test_implementation_only_import_main_executable(self):
        """Test `@_implementationOnly import` in the main executable

        See the ReadMe.md in the parent directory for more information.
        """

        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"), self.launch_info())

        # This test is deliberately checking what the user will see, rather than
        # the structure provided by the Python API, in order to test the recovery.
        self.expect("fr var", substrs=[
            "(SomeLibrary.TwoInts) value = (first = 2, second = 3)",
            "(main.ContainsTwoInts) container = {\n  wrapped = (first = 2, second = 3)\n  other = 10\n}",
            "(Int) simple = 1"])
        self.expect("e value", substrs=["(SomeLibrary.TwoInts)", "= (first = 2, second = 3)"])
        self.expect("e container", substrs=["(main.ContainsTwoInts)", "wrapped = (first = 2, second = 3)", "other = 10"])
        self.expect("e TwoInts(4, 5)", substrs=["(SomeLibrary.TwoInts)", "= (first = 4, second = 5)"])

    @skipIf(bugnumber="rdar://problem/54322424", # This test is unreliable.
            setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    @skipIfLinux # rdar://problem/67348391
    def test_implementation_only_import_main_executable_no_library_module(self):
        """Test `@_implementationOnly import` in the main executable, after removing the library's swiftmodule

        See the ReadMe.md in the parent directory for more information.
        """

        self.build()
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")
        os.remove(self.getBuildArtifact("SomeLibrary.swiftmodule"))
        os.remove(self.getBuildArtifact("SomeLibrary.swiftinterface"))
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"), self.launch_info())

        # FIXME: This particular test config is producing different results on
        # different machines, but it's also the least important configuration
        # (trying to debug something built against a library without that
        # library present, implementation-only or not, and that library doesn't
        # even have library evolution support enabled. Just make sure we don't
        # crash.
        self.expect("fr var", substrs=[
            "value = <could not resolve type>",
#            "container = {}",
            "simple = 1"])

        self.expect("e value", error=True)
        self.expect("e container", error=True)
        self.expect("e TwoInts(4, 5)", error=True)
        lldb.SBDebugger.MemoryPressureDetected()
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")

    @swiftTest
    @expectedFailureAll(oslist=["windows"])
    def test_implementation_only_import_main_executable_resilient(self):
        """Test `@_implementationOnly import` in the main executable with a resilient library

        See the ReadMe.md in the parent directory for more information.
        """

        self.build(dictionary={"LIBRARY_SWIFTFLAGS_EXTRAS": "-enable-library-evolution"})
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"), self.launch_info())

        # This test is deliberately checking what the user will see, rather than
        # the structure provided by the Python API, in order to test the recovery.
        self.expect("fr var", substrs=[
            "(SomeLibrary.TwoInts) value = (first = 2, second = 3)",
            "(main.ContainsTwoInts) container = {\n  wrapped = (first = 2, second = 3)\n  other = 10\n}",
            "(Int) simple = 1"])
        self.expect("e value", substrs=["(SomeLibrary.TwoInts)", "= (first = 2, second = 3)"])
        self.expect("e container", substrs=["(main.ContainsTwoInts)", "wrapped = (first = 2, second = 3)", "other = 10"])
        self.expect("e TwoInts(4, 5)", substrs=["(SomeLibrary.TwoInts)", "= (first = 4, second = 5)"])

    @swiftTest
    @expectedFailureOS(no_match(["macosx"])) # Requires Remote Mirrors support
    def test_implementation_only_import_main_executable_resilient_no_library_module(self):
        """Test `@_implementationOnly import` in the main executable with a resilient library, after removing the library's swiftmodule

        See the ReadMe.md in the parent directory for more information.
        """

        self.build(dictionary={"LIBRARY_SWIFTFLAGS_EXTRAS": "-enable-library-evolution"})
        os.remove(self.getBuildArtifact("SomeLibrary.swiftmodule"))
        os.remove(self.getBuildArtifact("SomeLibrary.swiftinterface"))
        lldbutil.run_to_source_breakpoint(self, "break here", lldb.SBFileSpec("main.swift"), self.launch_info())

        # This test is deliberately checking what the user will see, rather than
        # the structure provided by the Python API, in order to test the recovery.
        self.expect("fr var", substrs=[
            "value = <could not resolve type>",
            "(main.ContainsTwoInts) container = (other = 10)",
            "(Int) simple = 1"])
        # FIXME: If we could figure out how to ignore this failure but still not
        # crash if we touch something that can't be loaded, that would be nice.
        self.expect("e value", error=True, substrs=["failed to get module \"SomeLibrary\" from AST context"])
        self.expect("e container", error=True, substrs=["failed to get module \"SomeLibrary\" from AST context"])
        self.expect("e TwoInts(4, 5)", error=True, substrs=["failed to get module \"SomeLibrary\" from AST context"])
