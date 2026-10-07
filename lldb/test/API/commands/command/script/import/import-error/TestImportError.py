"""Check that we handle an ImportError in a special way when command script importing files."""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class ImportErrorTestCase(TestBase):
    @add_test_categories(["pyapi"])
    @no_debug_info_test
    def test_import_error_command(self):
        """Check that we handle an ImportError in a special way when command script importing files."""
        self.run_test()

    def run_test(self):
        """Check that we handle an ImportError in a special way when command script importing files."""

        self.expect(
            "command script import ./fail_importerror.py --allow-reload",
            error=True,
            substrs=['raise ImportError("I do not want to be imported")'],
        )
        self.expect(
            "command script import ./fail_valueerror.py --allow-reload",
            error=True,
            substrs=['raise ValueError("I do not want to be imported")'],
        )
