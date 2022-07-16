# TestPlaygrounds.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------
"""
Test that playgrounds work
"""
import subprocess
import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import os.path
import platform
import unittest2
from lldbsuite.test.builders.darwin import get_triple

import sys
if sys.version_info.major == 2:
    import commands as subprocess
else:
    import subprocess


def execute_command(command):
    (exit_status, output) = subprocess.getstatusoutput(command)
    return exit_status


class TestSwiftPlaygrounds(TestBase):

    def get_build_triple(self):
        """We want to build the file with a deployment target earlier than the
           availability set in the source file."""
        if lldb.remote_platform:
            arch = self.getArchitecture()
            vendor, os, version, _ = get_triple()
            # This is made slightly more complex by watchOS having misaligned
            # version numbers.
            if os == 'watchos':
                version = '5.0'
            else:
                version = '7.0'
            triple = '{}-{}-{}{}'.format(arch, vendor, os, version)
        else:
            triple = '{}-apple-macosx11.0'.format(platform.machine())
        return triple

    def get_run_triple(self):
        if lldb.remote_platform:
            arch = self.getArchitecture()
            vendor, os, version, _ = get_triple()
            triple = '{}-{}-{}{}'.format(arch, vendor, os, version)
        else:
            version, _, machine = platform.mac_ver()
            triple = '{}-apple-macosx{}'.format(machine, version)
        return triple

    @skipUnlessDarwin
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(debug_info=decorators.no_match("dsym"))
    def test_cross_module_extension(self):
        """Test that playgrounds work"""
        self.build(dictionary={
            'TARGET_SWIFTFLAGS':
            '-target {}'.format(self.get_build_triple()),
        })
        self.do_test(True)
        self.do_test(False)

    def setUp(self):
        TestBase.setUp(self)
        self.PlaygroundStub_source = "PlaygroundStub.swift"
        self.PlaygroundStub_source_spec = lldb.SBFileSpec(
            self.PlaygroundStub_source)

    def do_test(self, force_target):
        """Test that playgrounds work"""
        exe_name = "PlaygroundStub"
        exe = self.getBuildArtifact(exe_name)

        # Create the target
        if force_target:
            target = self.dbg.CreateTargetWithFileAndArch(
                exe, self.get_run_triple())
        else:
            target = self.dbg.CreateTarget(exe)

        self.assertTrue(target, VALID_TARGET)
        self.registerSharedLibrariesWithTarget(target,
                                               ['libPlaygroundsRuntime.dylib'])

        # Set the breakpoints
        breakpoint = target.BreakpointCreateBySourceRegex(
            'Set breakpoint here', self.PlaygroundStub_source_spec)
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertEqual(len(threads), 1)
        self.expect('settings set target.swift-framework-search-paths "%s"' %
                    self.getBuildDir())

        contents = ""

        with open('Contents.swift', 'r') as contents_file:
            contents = contents_file.read()

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)
        options.SetPlaygroundTransformEnabled()

        self.frame().EvaluateExpression(contents, options)
        ret = self.frame().EvaluateExpression("get_output()")
        playground_output = ret.GetSummary()
        if not force_target:
            # This is expected to fail because the deployment target
            # is less than the availability of the function being
            # called.
            self.assertEqual(playground_output, '""')
            return

        self.assertTrue(playground_output is not None)
        self.assertTrue("a=\\'3\\'" in playground_output)
        self.assertTrue("b=\\'5\\'" in playground_output)
        self.assertTrue("=\\'8\\'" in playground_output)
        self.assertTrue("=\\'11\\'" in playground_output)

        # Test concurrency
        contents = "error"
        with open('Concurrency.swift', 'r') as contents_file:
            contents = contents_file.read()
        res = self.frame().EvaluateExpression(contents, options)
        ret = self.frame().EvaluateExpression("get_output()")
        playground_output = ret.GetSummary()
        self.assertTrue(playground_output is not None)
        self.assertIn("=\\'23\\'", playground_output)

        # Test importing a library that adds new Clang options.
        log = self.getBuildArtifact('types.log')
        self.expect('log enable lldb types -f ' + log)
        contents = "import Dylib\nf()\n"
        res = self.frame().EvaluateExpression(contents, options)
        ret = self.frame().EvaluateExpression("get_output()")
        playground_output = ret.GetSummary()
        self.assertTrue(playground_output is not None)
        self.assertIn("Hello from the Dylib", playground_output)

        # Scan through the types log to make sure the SwiftASTContext was poisoned.
        import io
        logfile = io.open(log, "r", encoding='utf-8')
        found = 0
        for line in logfile:
            if 'New Swift image added' in line \
               and 'Versions/A/Dylib' in line \
               and 'ClangImporter needs to be reinitialized' in line:
                    found += 1
        self.assertEqual(found, 2)
