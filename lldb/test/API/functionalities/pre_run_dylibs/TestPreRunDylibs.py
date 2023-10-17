import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class TestPreRunLibraries(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(oslist=no_match(["darwin", "macos"]))
    def test(self):
        """Test that we find directly linked dylib pre-run."""

        self.build()
        target = self.dbg.CreateTarget(self.getBuildArtifact("a.out"))
        self.assertTrue(target, VALID_TARGET)

        # I don't know what the name of a shared library
        # extension is in general, so instead of using FindModule,
        # I'll iterate through the module and do a basename match.
        found_it = False
        for module in target.modules:
            file_name = module.GetFileSpec().GetFilename()
            if file_name.find("unlikely_name") != -1:
                found_it = True
                break

        self.assertTrue(
            found_it, "Couldn't find unlikely_to_occur_name in loaded libraries."
        )
