# TestSwiftRewriteClangPaths.py
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

import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import os
import shutil

class TestSwiftRewriteClangPaths(TestBase):
    # testWithoutRemap deletes the remapping plist out of the .dSYM, which make
    # does not regenerate, so the test functions cannot share a build directory.
    SHARED_BUILD_TESTCASE = False

    @requireNotEmbeddedSwift
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym"]))
    def testWithRemap(self):
        self.dotest(True)

    @requireNotEmbeddedSwift
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym"]))
    def testWithoutRemap(self):
        self.dotest(False)

    @skipEmbeddedSwift
    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    @skipIf(debug_info=no_match(["dsym"]))
    def testTargetSourceMapWins(self):
        """Document which remapping takes effect when the module's dSYM and
        target.source-map both match the same prefix: target.source-map is
        applied and the dSYM's DBGSourcePathRemapping is not, even though the
        dSYM's is the one that points at the real headers."""
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)
        self.expect("settings set symbols.swift-typesystem-compiler-fallback true")

        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")

        botdir = os.path.realpath(self.getBuildArtifact("buildbot"))
        userdir = os.path.realpath(self.getBuildArtifact("user"))
        self.assertTrue(os.path.isfile(self.find_plist()))

        # The dSYM's DBGSourcePathRemapping maps botdir onto userdir, which is
        # where the headers really are. Aim target.source-map at an empty
        # directory instead, so that it is observable which of the two is
        # applied. The directory has to exist because target.source-map
        # rejects a replacement path that does not.
        decoydir = os.path.realpath(self.getBuildArtifact("decoy"))
        lldbutil.mkdir_p(decoydir)
        self.runCmd("settings set target.source-map %s %s" % (botdir, decoydir))

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Foo.swift'),
            extra_images=['Foo'])

        # Because target.source-map is what gets applied, the include paths
        # end up in the empty decoy directory and the module cannot be built.
        self.expect("expression foo", error=True)
        self.filecheck_log(log, __file__, "--check-prefix=CHECK_DECOY")

        # Nothing was remapped onto the dSYM's destination.
        with open(log) as f:
            remaps = [l for l in f.read().splitlines() if "remapped " in l]
        self.assertTrue(remaps, "the ClangImporter paths were remapped")
        self.assertEqual(
            [l for l in remaps if userdir in l.split("->")[-1]], [],
            "the dSYM's remapping must not have been applied")

    def find_plist(self):
        import glob
        plist = self.getBuildArtifact("libFoo.dylib.dSYM/Contents/Resources/*.plist")
        lst = glob.glob(plist)
        self.assertTrue(len(lst) == 1)
        return lst[0]
        
    def dotest(self, remap):
        self.build()
        log = self.getBuildArtifact("types.log")
        self.runCmd('log enable lldb types -f "%s"' % log)

        # Because the bridging header isn't precompiled or in a module
        # we don't have DWARF type information for the types it contains.
        self.expect("settings set symbols.swift-typesystem-compiler-fallback true")

        # To ensure the module is rebuilt remove the cache to avoid caching.
        mod_cache = self.getBuildArtifact("my-clang-modules-cache")
        if os.path.isdir(mod_cache):
          shutil.rmtree(mod_cache)
        self.runCmd('settings set symbols.clang-modules-cache-path "%s"'
                    % mod_cache)
        self.runCmd("settings set symbols.use-swift-dwarfimporter false")

        botdir = os.path.realpath(self.getBuildArtifact("buildbot"))
        userdir = os.path.realpath(self.getBuildArtifact("user"))
        self.assertFalse(os.path.isdir(botdir))
        self.assertTrue(os.path.isdir(userdir))
        plist = self.find_plist()
        self.assertTrue(os.path.isfile(plist))
        if remap:
            self.runCmd("settings set target.source-map %s %s %s %s" %
                        (botdir, userdir, '/nonexisting-rootdir', userdir))
        else:
            # Also delete the remapping plist from the .dSYM to verify
            # that this doesn't work by happy accident without it.
            os.remove(plist)

        # Create the target
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('Foo.swift'),
            extra_images=['Foo'])

        if remap:
            comment = "returns correct value"
            self.expect("expression foo", comment, substrs=["x", "23"])
            self.expect("expression bar", comment, substrs=["y", "42"])
            self.expect("fr var foo", comment, substrs=["x", "23"])
            self.expect("fr var bar", comment, substrs=["y", "42"])
            self.assertTrue(os.path.isdir(mod_cache), "module cache exists")
        else:
            self.expect("expression foo", error=True)

        # Scan through the types log.
        suffix = "REMAP" if remap else "NORMAL"
        self.filecheck_log(log, __file__, "--check-prefix=CHECK_" + suffix)


# CHECK_REMAP-NOT: remapped -iquote
# CHECK_REMAP-NOT: error:{{.*}}Foo
# CHECK_NORMAL: error:{{.*}}Foo
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/Foo{{.*}} -> {{.*}}/user/Foo
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/iquote-path{{.*}} -> {{.*}}/user/iquote-path
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/I-double{{.*}} -> {{.*}}/user/I-double
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/I-single{{.*}} -> {{.*}}/user/I-single
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/Frameworks{{.*}} -> {{.*}}/user/Frameworks
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/nonexisting-rootdir{{.*}} -> {{.*}}/user
# CHECK_REMAP-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/Foo/overlay.yaml{{.*}} -> {{.*}}/user/Foo/overlay.yaml

# CHECK_DECOY-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/Foo{{.*}} -> {{.*}}/decoy/Foo
# CHECK_DECOY-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/I-single{{.*}} -> {{.*}}/decoy/I-single
# CHECK_DECOY-DAG: SwiftASTContextForExpressions(module: "Foo"{{.*}}/buildbot/Frameworks{{.*}} -> {{.*}}/decoy/Frameworks
