# TestSwiftDWARFImporterBridgingHeader.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2019 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os
import unittest2


class TestSwiftDWARFImporterBridgingHeader(lldbtest.TestBase):

    def build(self):
        include = self.getBuildArtifact('include')
        inputs = self.getSourcePath('Inputs')
        lldbutil.mkdir_p(include)
        import shutil
        for f in ['c-header.h']:
            shutil.copyfile(os.path.join(inputs, f), os.path.join(include, f))

        super(TestSwiftDWARFImporterBridgingHeader, self).build()

        # Remove the header files to prevent ClangImporter loading the
        # bridging header from source.
        self.assertTrue(os.path.isdir(include))
        shutil.rmtree(include)

    @skipIf(archs=['ppc64le'], bugnumber='SR-10214')
    # This test needs a working Remote Mirrors implementation.
    @skipIf(oslist=['windows'])
    # We delete the pch that would contains the debug info as part of the setup.
    #@skipIf(debug_info=no_match(["dsym"]))
    @swiftTest
    def test_dwarf_importer(self):
        lldb.SBDebugger.MemoryPressureDetected()
        self.runCmd("settings set symbols.use-swift-dwarfimporter true")
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("pureSwift"),
                                value="42")
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("point"),
                                typename='bridging-header.h.Point', num_children=2)
        self.expect("ta v -d no-dyn point", substrs=["x = 1", "y = 2"])
        self.expect("ta v -d no-dyn swiftStructCMember",
                    substrs=[
                        # FIXME: This doesn't even work with the original bridging header!
                        #"point", "x = 3", "y = 4",
                        "swift struct c member"])
        process.Clear()
        target.Clear()
        lldb.SBDebugger.MemoryPressureDetected()
