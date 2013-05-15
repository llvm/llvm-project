"""
Test that variables of integer basic types are displayed correctly.
"""

import AbstractBase
import unittest2
import sys
import lldb
from lldbtest import *

# rdar://problem/9649573
# Capture the lldb and gdb-remote log files for test failures when run with no "-w" option
class DebugIntegerTypesFailures(AbstractBase.GenericTester):

    mydir = "types"

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # If we're lucky, test_long_type_with_dsym fails.
        # Let's turn on logging just for that.
        try:
            if "test_long_type_with_dsym" in self.id():
                self.runCmd(
                    "log enable -n -f %s lldb commands event process state" %
                    os.environ["DEBUG_LLDB_LOG"])
                self.runCmd(
                    "log enable -n -f %s gdb-remote packets process" %
                    os.environ["DEBUG_GDB_REMOTE_LOG"])
        except:
            pass

    def tearDown(self):
        # Call super's tearDown().
        TestBase.tearDown(self)
        # If we're lucky, test_long_type_with_dsym fails.
        # Let's turn off logging just for that.
        if "test_long_type_with_dsym" in self.id():
            self.runCmd("log disable lldb")
            self.runCmd("log disable gdb-remote")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_char_type_with_dsym(self):
        """Test that char-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'char.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.char_type()

    def test_char_type_with_dwarf(self):
        """Test that char-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'char.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.char_type()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_short_type_with_dsym(self):
        """Test that short-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'short.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.short_type()

    def test_short_type_with_dwarf(self):
        """Test that short-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'short.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.short_type()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_int_type_with_dsym(self):
        """Test that int-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'int.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type()

    def test_int_type_with_dwarf(self):
        """Test that int-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'int.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.int_type()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_long_type_with_dsym(self):
        """Test that long-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long.cpp'}
        print self.id()
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type()

    def test_long_type_with_dwarf(self):
        """Test that long-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_type()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    def test_long_long_type_with_dsym(self):
        """Test that 'long long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long_long.cpp'}
        self.buildDsym(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_long_type()

    def test_long_long_type_with_dwarf(self):
        """Test that 'long long'-type variables are displayed correctly."""
        d = {'CXX_SOURCES': 'long_long.cpp'}
        self.buildDwarf(dictionary=d)
        self.setTearDownCleanup(dictionary=d)
        self.long_long_type()

    def char_type(self):
        """Test that char-type variables are displayed correctly."""
        self.generic_type_tester(set(['char']), quotedDisplay=True)

    def int_type(self):
        """Test that int-type variables are displayed correctly."""
        self.generic_type_tester(set(['int']))

    def short_type(self):
        """Test that short-type variables are displayed correctly."""
        self.generic_type_tester(set(['short']))

    def long_type(self):
        """Test that long-type variables are displayed correctly."""
        self.generic_type_tester(set(['long']))

    def long_long_type(self):
        """Test that long long-type variables are displayed correctly."""
        self.generic_type_tester(set(['long long']))


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
