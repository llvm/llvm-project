"""
Test that lldb command "command source" works correctly.

See also http://llvm.org/viewvc/llvm-project?view=rev&revision=109673.
"""

import os, sys
import unittest2
import lldb
from lldbtest import *

class CommandSourceTestCase(TestBase):

    mydir = os.path.join("functionalities", "command_source")

    def test_command_source(self):
        """Test that lldb command "command source" works correctly."""

        # Sourcing .lldb in the current working directory, which in turn imports
        # the "my" package that defines the date() function.
        self.runCmd("command source .lldb")

        # Let's temporarily redirect the stdout to our StringIO session object
        # in order to capture the script evaluation output.
        old_stdout = sys.stdout
        session = StringIO.StringIO()
        sys.stdout = session

        # Python should evaluate "my.date()" successfully.
        # Pass 'check=False' so that sys.stdout gets restored unconditionally.
        self.runCmd("script my.date()", check=False)

        # Now restore stdout to the way we were. :-)
        sys.stdout = old_stdout

        import datetime
        self.expect(session.getvalue(), "script my.date() runs successfully",
                    exe=False,
            substrs = [str(datetime.date.today())])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
