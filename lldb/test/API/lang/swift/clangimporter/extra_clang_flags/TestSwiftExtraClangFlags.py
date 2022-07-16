import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil
import unittest2

class TestSwiftExtraClangFlags(TestBase):

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(oslist=['windows'])
    @swiftTest
    def test_sanity(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("frame var foo", "sanity check", substrs=['(Foo)'])
        self.expect("expr FromOverlay(i: 23)", error=True)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(oslist=['windows'])
    @swiftTest
    def test_extra_clang_flags(self):
        """
        Test that a debug-only module map can be installed by injecting a
        VFS overlay using target.swift-extra-clang-flags.
        """
        self.build()
        # FIXME: this doesn't work if LLDB's build dir contains a space.
        overlay = self.getBuildArtifact('overlay.yaml')
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.swift-extra-clang-flags"))
        self.expect('settings set -- target.swift-extra-clang-flags '+
                    '"-ivfsoverlay %s"' % overlay)
        with open(overlay, 'w+') as f:
            import os
            f.write("""
{
  'version': 0,
  'roots': [
    { 'name': '"""+os.getcwd()+"""/nonmodular', 'type': 'directory',
      'contents': [
        { 'name': 'module.modulemap', 'type': 'file',
          'external-contents': '"""+os.path.join(os.getcwd(),
                                                 'overlaid.map')+"""'
        }
      ]
    }
  ]
}
""")
        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("frame var foo", "sanity check", substrs=['(Foo)'])
        self.expect("expr FromOverlay(i: 23)", substrs=['(FromOverlay)', '23'])

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipIf(oslist=['windows'])
    @swiftTest
    def test_invalid_extra_clang_flags(self):
        """
        Test that LLDB ignores specific invalid arguments in
        swift-extra-clang-flags.
        """
        self.build()
        self.addTearDownHook(
            lambda: self.runCmd("settings clear target.swift-extra-clang-flags"))

        self.expect('settings set target.swift-extra-clang-flags -- -v')

        lldbutil.run_to_source_breakpoint(self, "break here",
                                          lldb.SBFileSpec('main.swift'))
        self.expect("frame var foo", substrs=['(Foo)'])
