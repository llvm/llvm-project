import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil
import os

class TestSwiftDWARFImporterFromDylib(lldbtest.TestBase):

    @swiftTest
    # This test needs a working Remote Mirrors implementation.
    @skipIf(oslist=['linux', 'windows'])
    def test_dwarf_importer(self):
        self.build()
        #os.remove(self.getBuildArtifact('Foo.swiftmodule'))
        lldbutil.run_to_source_breakpoint(self, 'break here',
                                          lldb.SBFileSpec('main.swift'),
                                          extra_images=['Foo'])
        # This type can only be imported into Swift via DWARFImporter
        # and is not visible from the main module at all.
        self.expect("expr -- WrappingFromDylib()", substrs=['23'])
