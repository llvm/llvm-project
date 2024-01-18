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
    def test_force_target(self):
        """Test that playgrounds work"""
        self.launch(True)
        self.do_basic_test(True)

    @skipUnlessDarwin
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(debug_info=decorators.no_match("dsym"))
    def test_no_force_target(self):
        """Test that playgrounds work"""
        self.launch(False)
        self.do_basic_test(False)

    @skipUnlessDarwin
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(debug_info=decorators.no_match("dsym"))
    @skipIf(macos_version=["<", "12"])
    def test_concurrency(self):
        """Test that concurrency is available in playgrounds"""
        self.launch(True)
        self.do_concurrency_test()

    @skipUnlessDarwin
    @swiftTest
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(debug_info=decorators.no_match("dsym"))
    def test_import(self):
        """Test that a dylib can be imported in playgrounds"""
        self.launch(True)
        self.do_import_test()
        
    def launch(self, force_target):
        """Test that playgrounds work"""
        self.build(dictionary={
            'TARGET_SWIFTFLAGS':
            '-target {}'.format(self.get_build_triple()),
        })

        # Create the target
        exe = self.getBuildArtifact("PlaygroundStub")
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
            'Set breakpoint here', lldb.SBFileSpec("PlaygroundStub.swift"))
        self.assertTrue(breakpoint.GetNumLocations() > 0, VALID_BREAKPOINT)

        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        threads = lldbutil.get_threads_stopped_at_breakpoint(
            process, breakpoint)

        self.assertEqual(len(threads), 1)
        self.expect('settings set target.swift-framework-search-paths "%s"' %
                    self.getBuildDir())

    def execute_code(self, input_file, expect_error=False):
        contents = "syntax error"
        with open(input_file, 'r') as contents_file:
            contents = contents_file.read()

        options = lldb.SBExpressionOptions()
        options.SetLanguage(lldb.eLanguageTypeSwift)
        options.SetPlaygroundTransformEnabled()
        # The concurrency expressions will spawn multiple threads.
        options.SetOneThreadTimeoutInMicroSeconds(1)
        options.SetTryAllThreads(True)
        options.SetAutoApplyFixIts(False)

        res = self.frame().EvaluateExpression(contents, options)
        ret = self.frame().EvaluateExpression("get_output()")
        is_error = res.GetError().Fail() and not (
                     res.GetError().GetType() == 1 and
                     res.GetError().GetError() == 0x1001)
        playground_output = ret.GetSummary()
        with recording(self, self.TraceOn()) as sbuf:
            print("playground result: ", file=sbuf)
            print(str(res), file=sbuf)
            if is_error:
                print("error:", file=sbuf)
                print(str(res.GetError()), file=sbuf)
            else:
                print("playground output:", file=sbuf)
                print(str(ret), file=sbuf)

        if expect_error:
            self.assertTrue(is_error)
            return playground_output
        self.assertFalse(is_error)
        self.assertIsNotNone(playground_output)
        return playground_output
        
    def do_basic_test(self, force_target):
        playground_output = self.execute_code('Contents.swift', not force_target)
        if not force_target:
            # This is expected to fail because the deployment target
            # is less than the availability of the function being
            # called.
            self.assertEqual(playground_output, '""')
            return

        self.assertIn("a=\\'3\\'", playground_output)
        self.assertIn("b=\\'5\\'", playground_output)
        self.assertIn("=\\'8\\'", playground_output)
        self.assertIn("=\\'11\\'", playground_output)

    def do_concurrency_test(self):
        playground_output = self.execute_code('Concurrency.swift')
        self.assertIn("=\\'23\\'", playground_output)

    def do_import_test(self):
        # Test importing a library that adds new Clang options.
        log = self.getBuildArtifact('types.log')
        self.expect('log enable lldb types -f ' + log)
        playground_output = self.execute_code('Import.swift')
        self.assertIn("Hello from the Dylib", playground_output)

        # Scan through the types log to make sure the SwiftASTContext was poisoned.
        self.filecheck('platform shell cat ""%s"' % log, __file__)
#       CHECK: New Swift image added{{.*}}Versions/A/Dylib{{.*}}ClangImporter needs to be reinitialized
