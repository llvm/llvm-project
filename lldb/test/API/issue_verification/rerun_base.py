
import os

import lldbsuite.test.lldbtest as lldbtest


# pylint: disable=too-few-public-methods
class RerunBaseTestCase(lldbtest.TestBase):
    """Forces test failure."""
    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def should_generate_issue(self):
        """Returns whether a test issue should be generated.

        @returns True on the first and every other call via a given
        test method.
        """
        should_pass_filename = "{}.{}.succeed-marker".format(
            __file__, self.id())
        fail = not os.path.exists(should_pass_filename)
        if fail:
            # Create the marker so that next call to this passes.
            open(should_pass_filename, 'w').close()
        else:
            # Delete the marker so next time we fail.
            os.remove(should_pass_filename)
        return fail
