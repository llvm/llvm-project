from __future__ import print_function
import lldbsuite.test.lldbplaygroundrepl as repl
from lldbsuite.test.lldbtest import *

class TestPlaygroundREPLImport(repl.PlaygroundREPLTest):

    mydir = repl.PlaygroundREPLTest.compute_mydir(__file__)
    
    def do_test(self):
        """
        Test importing a library that adds new Clang options.
        """
        self.expect('settings set target.swift-framework-search-paths "%s"' %
                    self.getBuildDir())
        self.expect('settings set target.use-all-compiler-flags true')

        log = self.getBuildArtifact('types.log')
        self.expect('log enable lldb types -f ' + log)
        result, playground_output = self.execute_code('BeforeImport.swift')
        self.assertIn("persistent", playground_output.GetSummary())
        result, playground_output = self.execute_code('Import.swift')
        self.assertIn("Hello from the Dylib", playground_output.GetSummary())
        self.assertIn("and back again", playground_output.GetSummary())

        # Scan through the types log to make sure the SwiftASTContext was poisoned.
        self.filecheck('platform shell cat ""%s"' % log, __file__)
#       CHECK: New Swift image added{{.*}}Versions/A/Dylib{{.*}}ClangImporter needs to be reinitialized
