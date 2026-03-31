"""
Test that variables of integer basic types are displayed correctly.
"""


import AbstractBase
import lldb
from lldbsuite.test.lldbtest import *

# rdar://problem/9649573
# Capture the lldb and gdb-remote log files for test failures when run
# with no "-w" option


class DebugIntegerTypesFailures(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # If we're lucky, test_long_type_with_dsym fails.
        # Let's turn on logging just for that.
        try:
            if "test_long_type_with_dsym" in self.id():
                self.runCmd(
                    "log enable -n -f %s lldb commands event process state"
                    % os.environ["DEBUG_LLDB_LOG"]
                )
                self.runCmd(
                    "log enable -n -f %s gdb-remote packets process"
                    % os.environ["DEBUG_GDB_REMOTE_LOG"]
                )
        except:
            pass

    def tearDown(self):
        # If we're lucky, test_long_type_with_dsym fails.
        # Let's turn off logging just for that.
        if "test_long_type_with_dsym" in self.id():
            self.runCmd("log disable lldb")
            self.runCmd("log disable gdb-remote")
        # Call super's tearDown().
        TestBase.tearDown(self)

    def test_char_type(self):
        """Test that char-type variables are displayed correctly."""
        d = {"CXX_SOURCES": "char.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.generic_type_tester(set(["char"]), quotedDisplay=True)

    def test_short_type(self):
        """Test that short-type variables are displayed correctly."""
        d = {"CXX_SOURCES": "short.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.generic_type_tester(set(["short"]))

    def test_int_type(self):
        """Test that int-type variables are displayed correctly."""
        d = {"CXX_SOURCES": "int.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.generic_type_tester(set(["int"]))

    def test_long_type(self):
        """Test that long-type variables are displayed correctly."""
        d = {"CXX_SOURCES": "long.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.generic_type_tester(set(["long"]))

    def test_long_long_type(self):
        """Test that 'long long'-type variables are displayed correctly."""
        d = {"CXX_SOURCES": "long_long.cpp"}
        self.build(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.generic_type_tester(set(["long long"]))
