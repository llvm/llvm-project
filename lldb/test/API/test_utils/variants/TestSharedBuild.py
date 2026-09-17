import os

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestCase(TestBase):
    SHARED_BUILD_TESTCASE = True

    # Maps debug_info variant to mtime of the built executable.
    # Shared across test methods.
    _exe_mtimes = {}

    def assertSharedBuild(self):
        self.build()

        # Check that the debug info variant matches the build binaries.
        debug_info = self.getDebugInfo()
        dsym_path = self.getBuildArtifact("a.out.dSYM")
        if debug_info == "dsym":
            self.assertTrue(
                os.path.isdir(dsym_path),
                "the dsym variant should build a .dSYM bundle",
            )
        elif debug_info == "dwarf":
            self.assertFalse(
                os.path.exists(dsym_path),
                "the dwarf variant should not build a .dSYM bundle",
            )
        else:
            self.fail(f"Unknown debug info kind: {debug_info}")

        # Check that we didn't rebuild the source files between the shared
        # test folders.
        exe_mtime = os.path.getmtime(self.getBuildArtifact("a.out"))
        if debug_info in TestCase._exe_mtimes:
            self.assertEqual(
                exe_mtime,
                TestCase._exe_mtimes[debug_info],
                "shared test case rebuilt test sources?",
            )
        else:
            TestCase._exe_mtimes[debug_info] = exe_mtime

    @add_test_categories(["dwarf", "dsym"])
    def test_one_builds(self):
        self.assertSharedBuild()

    @add_test_categories(["dwarf", "dsym"])
    def test_two_reuses_build(self):
        self.assertSharedBuild()
