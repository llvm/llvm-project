import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os


class TestSwiftImportSPI(lldbtest.TestBase):
    @swiftTest
    def test(self):
        """Test SPI imports"""
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.swift'),
            extra_images=['A', 'B'])
        # a.b returns a B FromB, which is an @_spi(Private) import of A.
        self.expect("expression -- a.b.c", substrs=['42'])
        # a.a itself is marked @_spi(Private).
        self.expect("expression -- a.a", substrs=['42'])
