from __future__ import absolute_import

import os

import lit.Test
import lit.TestRunner
from .base import TestFormat

class ShTest(TestFormat):
    """ShTest is a format with one file per test.

    This is the primary format for regression tests as described in the LLVM
    testing guide:

        http://llvm.org/docs/TestingGuide.html

    The ShTest files contain some number of shell-like command pipelines, along
    with assertions about what should be in the output.
    """

    def __init__(self, execute_external = False):
        """Initializer.

        The 'execute_external' argument controls whether lit uses its internal
        logic for command pipelines, or passes the command to a shell
        subprocess.

        Args:
            execute_external: (optional) If true, use shell subprocesses instead
                of lit's internal pipeline logic.
        """
        self.execute_external = execute_external

    def getTestsInDirectory(self, testSuite, path_in_suite,
                            litConfig, localConfig):
        """Yields test files matching 'suffixes' from the localConfig."""
        source_path = testSuite.getSourcePath(path_in_suite)
        for filename in os.listdir(source_path):
            # Ignore dot files and excluded tests.
            if (filename.startswith('.') or
                filename in localConfig.excludes):
                continue

            filepath = os.path.join(source_path, filename)
            if not os.path.isdir(filepath):
                base,ext = os.path.splitext(filename)
                if ext in localConfig.suffixes:
                    yield lit.Test.Test(testSuite, path_in_suite + (filename,),
                                        localConfig)

    def execute(self, test, litConfig):
        """Interprets and runs the given test file, and returns the result."""
        return lit.TestRunner.executeShTest(test, litConfig,
                                            self.execute_external)
