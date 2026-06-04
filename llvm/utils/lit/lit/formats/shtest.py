from __future__ import annotations

import lit.TestRunner
import lit.util

from .base import FileBasedTest
from typing import TYPE_CHECKING

# Can't import unconditionally due to circular dependency
if TYPE_CHECKING:
    from lit.Test import Test, Result
    from lit.LitConfig import LitConfig


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
        execute_external: bool = False,
        extra_substitutions: list[tuple[str, str]] = [],
        preamble_commands: list[str] = [],
    ) -> None:
        self.execute_external = execute_external
        self.extra_substitutions = extra_substitutions
        self.preamble_commands = preamble_commands

    def execute(self, test: Test, litConfig: LitConfig) -> Result:
        return lit.TestRunner.executeShTest(
            test,
            litConfig,
            self.execute_external,
            self.extra_substitutions,
            self.preamble_commands,
        )
