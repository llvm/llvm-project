import lit.TestRunner
import lit.util

from .base import FileBasedTest


class ShTest(FileBasedTest):
    """ShTest is a format with one file per test.

    This is the primary format for regression tests as described in the LLVM
    testing guide:

        http://llvm.org/docs/TestingGuide.html

    The ShTest files contain some number of shell-like command pipelines, along
    with assertions about what should be in the output.
    """

    def __init__(
        self,
        execute_external=False,
        extra_substitutions=[],
        preamble_commands=[],
        force_execute_external=False,
    ):
        if execute_external and not force_execute_external:
            raise ValueError(
                "execute_external=True is deprected as of LLVM-23 and the option will "
                "be removed in LLVM-24. Please move to using the internal shell "
                "(execute_external=False). If you still need to force external "
                "execution to allow time for migration, set force_execute_external=True"
            )
        self.execute_external = execute_external
        self.extra_substitutions = extra_substitutions
        self.preamble_commands = preamble_commands

    def execute(self, test, litConfig):
        return lit.TestRunner.executeShTest(
            test,
            litConfig,
            self.execute_external,
            self.extra_substitutions,
            self.preamble_commands,
        )
