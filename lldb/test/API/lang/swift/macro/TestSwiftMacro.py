import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os

class TestSwiftMacro(lldbtest.TestBase):

    NO_DEBUG_INFO_TESTCASE = True

    def setupPluginServerForTesting(self):
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


    @swiftTest
    # At the time of writing swift/test/Macros/macro_expand.swift is also disabled.
    @expectedFailureAll(oslist=["linux"])
    def testDebugging(self):
        """Test Swift macros"""
        self.build(dictionary={'SWIFT_SOURCES': 'main.swift'})
        self.setupPluginServerForTesting()
        main_spec = lldb.SBFileSpec('main.swift')
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', main_spec
        )

        # We're testing line breakpoint setting here:
        call_site_line = lldbtest.line_number("main.swift", "#no_return(a / b)")
        call_site_bp = target.BreakpointCreateByLocation(main_spec, call_site_line)

        thread.StepOver()
        thread.StepInto()
        
        # This is the expanded macro source, we should be able to step into it.
        # Don't check the actual line number so we are line independent
        self.assertIn(
            'freestanding macro expansion #1 of stringify in module a file main.swift line 5 column 11',
            thread.frames[0].name,
            "Stopped in stringify macro"
        )

        self.expect('expression -- #stringify(1)', substrs=['0 = 1', '1 = "1"'])

        # Step out should get us out of stringify, then in to the next macro:
        thread.StepOut()
        self.assertIn("a.testStringify", thread.frames[0].name, "Step out back to origin")
        thread.StepInto()
        self.assertIn(
            "freestanding macro expansion #1 of no_return in module a file main.swift line 6 column 3",
            thread.frames[0].name,
            "Step out and in gets to no_return"
        )
        
        # We've set a breakpoint on the call site for another instance - run to that:
        threads = lldbutil.continue_to_breakpoint(process, call_site_bp)
        self.assertEqual(len(threads), 1, "Stopped at one thread")
        thread = threads[0]
        frame_0 = thread.frames[0]
        line_entry_0 = frame_0.line_entry
        self.assertEqual(line_entry_0.line, call_site_line, "Got the right line attribution")
        self.assertEqual(line_entry_0.file, main_spec, "Got the right file attribution") 

        # Now test stepping in and back out again:
        thread.StepInto()
        self.assertIn(
            "freestanding macro expansion #3 of no_return in module a file main.swift line 8 column 3",
            thread.frames[0].name,
            "Step out and in gets to no_return"
        )
        
        thread.StepOut()
        self.assertIn("a.testStringify", thread.frames[0].name, "Step out from no_return")
        
        # Make sure we can set a symbolic breakpoint on a macro.
        b = target.BreakpointCreateByName("stringify")
        self.assertGreaterEqual(b.GetNumLocations(), 1)

    @swiftTest
    # At the time of writing swift/test/Macros/macro_expand.swift is also disabled.
    @expectedFailureAll(oslist=["linux"])
    def testInteractive(self):
        """Test Swift macros that are loaded via a user-initiated import"""
        self.build(dictionary={'SWIFT_SOURCES': 'empty.swift'})
        self.setupPluginServerForTesting()
        target, process, thread, bkpt = lldbutil.run_to_name_breakpoint(
            self, "main"
        )

        types_log = self.getBuildArtifact('types.log')
        self.expect('log enable lldb types -f "%s"' % types_log)
        self.expect('expression -- import Macro')
        self.expect('expression -- #stringify(1)', substrs=['0 = 1', '1 = "1"'])
        self.filecheck('platform shell cat "%s"' % types_log, __file__)
#       CHECK: CacheUserImports(){{.*}}: Macro.
#       CHECK: SwiftASTContextForExpressions{{.*}}::LoadOneModule(){{.*}}Imported module Macro from {kind = Serialized Swift AST, filename = "{{.*}}Macro.swiftmodule";}
#       CHECK: CacheUserImports(){{.*}}Scanning for search paths in{{.*}}Macro.swiftmodule
#       The bots have too old an Xcode for this.
#       DISABLED: SwiftASTContextForExpressions{{.*}}::LogConfiguration(){{.*}} -external-plugin-path {{.*}}/Developer/Platforms/{{.*}}.platform/Developer/usr/local/lib/swift/host/plugins{{.*}}#{{.*}}/swift-plugin-server
#       CHECK: SwiftASTContextForExpressions{{.*}}::LogConfiguration(){{.*}} -external-plugin-path {{.*}}/lang/swift/macro/{{.*}}#{{.*}}/swift-plugin-server
