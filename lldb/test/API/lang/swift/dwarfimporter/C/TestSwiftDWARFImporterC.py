# TestSwiftDWARFImporterC.py
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


class TestSwiftDWARFImporterC(lldbtest.TestBase):

    def build(self):
        include = self.getBuildArtifact('include')
        inputs = self.getSourcePath('Inputs')
        lldbutil.mkdir_p(include)
        import shutil
        for f in ['module.modulemap', 'c-header.h', 'submodule.h']:
            shutil.copyfile(os.path.join(inputs, f), os.path.join(include, f))

        super(TestSwiftDWARFImporterC, self).build()

        # Remove the header files to thwart ClangImporter.
        self.assertTrue(os.path.isdir(include))
        shutil.rmtree(include)

    @skipIf(archs=['ppc64le'], bugnumber='SR-10214')
    @swiftTest
    # This test needs a working Remote Mirrors implementation.
    @skipIf(oslist=['windows'])
    def test_dwarf_importer(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("pureSwift"),
                                value="42")
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("point"),
                                typename='CModule.Point', num_children=2)
        self.expect("target variable point", substrs=["x = 1", "y = 2"])
        self.expect("target variable enumerator", substrs=[".yellow"])
        self.expect("target variable pureSwiftStruct", substrs=["pure swift"])
        self.expect("target variable swiftStructCMember",
                    substrs=["point", "x = 3", "y = 4",
                             "sub", "x = 1", "y = 2", "z = 3",
                             "swift struct c member"])
        self.expect("target variable typedef", substrs=["x = 5", "y = 6"])
        #self.expect("target variable union",
        #            substrs=["(DoubleLongUnion)", "long_val = 42"])
        self.expect("target variable fromSubmodule",
                    substrs=["(FromSubmodule)", "x = 1", "y = 2", "z = 3"])

    @skipIf(archs=['ppc64le'], bugnumber='SR-10214')
    @swiftTest
    # This test needs a working Remote Mirrors implementation.
    @skipIf(oslist=['windows'])
    def test_dwarf_importer_exprs(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        self.expect("expr union", substrs=["(DoubleLongUnion)", "long_val = 42"])
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("pureSwift"),
                                value="42")
        lldbutil.check_variable(self,
                                target.FindFirstGlobalVariable("point"),
                                typename='CModule.Point', num_children=2)
        self.expect("expr point", substrs=["x = 1", "y = 2"])
        self.expect("expr enumerator", substrs=[".yellow"])
        self.expect("expr pureSwiftStruct", substrs=["pure swift"])
        self.expect("expr swiftStructCMember",
                    substrs=["point", "x = 3", "y = 4",
                             "sub", "x = 1", "y = 2", "z = 3",
                             "swift struct c member"])
        self.expect("expr typedef", substrs=["x = 5", "y = 6"])
        self.expect("expr union", substrs=["(DoubleLongUnion)", "long_val = 42"])
        self.expect("expr fromSubmodule",
                    substrs=["(FromSubmodule)", "x = 1", "y = 2", "z = 3"])
        
    @skipIf(archs=['ppc64le'], bugnumber='SR-10214')
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    def test_negative(self):
        lldb.SBDebugger.MemoryPressureDetected()
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))
        #lldbutil.check_variable(self,
        #                        target.FindFirstGlobalVariable("point"),
        #                        typename="Point", num_children=2)
        # This can't be resolved.
        self.expect("expr swiftStructCMember", error=True)

        found = False
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        for line in logfile:
            if "missing required module" in line:
                found = True
        self.assertTrue(found)

        process.Clear()
        target.Clear()
        lldb.SBDebugger.MemoryPressureDetected()
