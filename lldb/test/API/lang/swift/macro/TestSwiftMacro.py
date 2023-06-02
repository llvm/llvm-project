import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import unittest2
import os

class TestSwiftMacro(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    @swiftTest
    # At the time of writing swift/test/Macros/macro_expand.swift is also disabled.
    @expectedFailureAll(oslist=['linux'])
    def test(self):
        """Test Swift macros"""
        self.build()

        # Find the path to the just-built swift-plugin-server.
        # FIXME: this is not very robust.
        def replace_last(old, new, string):
            return new.join(string.rsplit(old, 1))

        swift_plugin_server = replace_last('clang', 'swift-plugin-server',
                                           self.getCompiler())
        if not os.path.exists(swift_plugin_server):
            swift_plugin_server = replace_last('llvm', 'swift',
                                               swift_plugin_server)
        self.assertTrue(os.path.exists(swift_plugin_server),
                        'could not find swift-plugin-server, tried "%s"'
                        %swift_plugin_server)
        self.runCmd(
            'settings set target.experimental.swift-plugin-server-for-path %s=%s'
            % (self.getBuildDir(), swift_plugin_server))

        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'))

        thread.StepOver()
        thread.StepInto()
        # This is the expanded macro source, we should be able to step into it.
        self.expect('reg read pc', substrs=[
            '[inlined] freestanding macro expansion #1 of stringify',
            '13testStringify'
        ])

        self.expect('expression -- #stringify(1)', substrs=['0 = 1', '1 = "1"'])

        # Make sure we can set a symbolic breakpoint on a macro.
        b = target.BreakpointCreateByName("stringify")
        self.assertGreaterEqual(b.GetNumLocations(), 1)
