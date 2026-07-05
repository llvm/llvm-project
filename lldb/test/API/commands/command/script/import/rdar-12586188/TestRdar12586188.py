"""Check that we handle an ImportError in a special way when command script importing files."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Rdar12586188TestCase(TestBase):
    @add_test_categories(["pyapi"])
    @no_debug_info_test
    def test_rdar12586188_command(self):
        """Check that we handle an ImportError in a special way when command script importing files."""
        self.run_test()

    def run_test(self):
        """Check that we handle an ImportError in a special way when command script importing files."""

        self.expect(
            "command script import ./fail12586188.py --allow-reload",
            error=True,
            substrs=['raise ImportError("I do not want to be imported")'],
        )
        self.expect(
            "command script import ./fail212586188.py --allow-reload",
            error=True,
            substrs=['raise ValueError("I do not want to be imported")'],
        )
