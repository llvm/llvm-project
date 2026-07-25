import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbutil as lldbutil

class TestSwiftSharedStringStorage(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipEmbeddedSwift
    @swiftTest
    @skipUnlessFoundation
    def test(self):
        """Non-ASCII strings bridged to NSString/CFString are backed by
        __SharedStringStorage. Check that the data formatter decodes them,
        which depends on locating the `start` field inside the storage
        object (the field whose offset changed when `_owner` was removed)."""
        self.build()

        logfile = self.getBuildArtifact("formatters.log")
        self.runCmd("log enable lldb formatters -v -f " + logfile)

        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'))
        self.expect("frame variable ns",
                    substrs=["café", "non-ASCII shared-storage NSString"])
        self.expect("frame variable cf",
                    substrs=["alçada", "non-ASCII shared-storage CFString"])

        with open(logfile) as f:
            log = f.read()
        self.assertIn("__SharedStringStorage `start` field at offset", log)
