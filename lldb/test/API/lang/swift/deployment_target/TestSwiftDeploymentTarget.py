# TestSwiftDeploymentTarget.py
#
# This source file is part of the Swift.org open source project
#
# Copyright (c) 2018 Apple Inc. and the Swift project authors
# Licensed under Apache License v2.0 with Runtime Library Exception
#
# See https://swift.org/LICENSE.txt for license information
# See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
#
# ------------------------------------------------------------------------------

import lldbsuite.test.lldbinline as lldbinline

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftDeploymentTarget(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "11.1"])
    @swiftTest
    def test_swift_deployment_target(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self,
                                          "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("expression f", substrs=['i = 23'])

    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "11.1"])
    @swiftTest
    def test_swift_deployment_target_dlopen(self):
        self.build()
        target, process, _, _, = lldbutil.run_to_name_breakpoint(
            self, 'main', exe_name="dlopen_module")
        bkpt = target.BreakpointCreateBySourceRegex(
            'break here', lldb.SBFileSpec('NewerTarget.swift'))
        lldbutil.continue_to_breakpoint(process, bkpt)
        self.expect("expression self", substrs=['i = 23'])

    @skipUnlessDarwin
    @skipIfDarwinEmbedded # This test uses macOS triples explicitly.
    @skipIf(macos_version=["<", "11.1"])
    # FIXME: This config started failing in CI only after switching to
    # the query-based FindTypes API.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @swiftTest
    def test_swift_deployment_target_from_macho(self):
        self.build(dictionary={"MAKE_DSYM": "NO"})
        os.unlink(self.getBuildArtifact("a.swiftmodule"))
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("main.swift")
        )
        self.expect("expression f", substrs=["i = 23"])
        self.filecheck('platform shell cat ""%s"' % log, __file__)
#       CHECK: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::SetTriple({{.*}}apple-macosx11.0.0
#       CHECK-NOT: SwiftASTContextForExpressions(module: "a", cu: "main.swift")::RegisterSectionModules("a.out"){{.*}} AST Data blobs

    @skipUnlessDarwin  # This test uses macOS triples explicitly.
    @skipIfDarwinEmbedded
    @skipIf(macos_version=["<", "11.1"])
    @skipIf(setting=("symbols.swift-precise-compiler-invocation", "false"))
    @swiftTest
    def test_swift_precise_compiler_invocation_triple(self):
        """
        Ensure expressions prefer the target triple of their module, as it may
        differ from the target triple of the target. This is necessary for
        explicitly built modules.
        """
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd(f'log enable lldb types -f "{log}"')
        lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec("NewerTarget.swift")
        )
        self.expect(
            "image list -t libNewerTarget.dylib",
            substrs=["-apple-macosx11.1.0"],
        )
        self.expect("expression self", substrs=["i = 23"])
        self.filecheck(
            f'platform shell cat "{log}"', __file__, "-check-prefix=CHECK-PRECISE"
        )
#       CHECK-PRECISE: SwiftASTContextForExpressions(module: "NewerTarget", cu: "NewerTarget.swift")::CreateInstance() -- Fully specified target triple {{.*}}-apple-macosx11.1.0
#       CHECK-PRECISE: SwiftASTContextForExpressions(module: "NewerTarget", cu: "NewerTarget.swift")::SetTriple("{{.*}}-apple-macosx11.1.0")
