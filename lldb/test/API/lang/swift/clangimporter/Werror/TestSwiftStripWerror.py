import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftWerror(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)

    # Don't run ClangImporter tests if Clangimporter is disabled.
    @skipIf(setting=('symbols.use-swift-clangimporter', 'false'))
    @skipUnlessDarwin
    @swiftTest
    def test(self):
        """This tests that -Werror is removed from ClangImporter options by
           introducing two conflicting macro definitions in idfferent dylibs.
        """
        self.build()
        target,  _, _, _ = lldbutil.run_to_source_breakpoint(
            self, "break here", lldb.SBFileSpec('dylib.swift'),
            extra_images=['Dylib'])

        # Turn on logging.
        log = self.getBuildArtifact("types.log")
        self.expect("log enable lldb types -f "+log)
        
        self.expect("p foo", DATA_TYPES_DISPLAYED_CORRECTLY, substrs=["42"])
        sanity = 0
        logfile = open(log, "r")
        for line in logfile:
            self.assertFalse("-Werror" in line)
            if "-DCONFLICT" in line:
                sanity += 1
        # We see it twice in the expression context and once in a Module context.
        #  -DCONFLICT=0
        #  -DCONFLICT=1
        self.assertEqual(sanity, 2+1)
