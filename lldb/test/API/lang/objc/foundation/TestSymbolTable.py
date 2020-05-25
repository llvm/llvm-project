"""
Test symbol table access for main.m.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


@skipUnlessDarwin
class FoundationSymtabTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    symbols_list = ['-[MyString initWithNSString:]',
                    '-[MyString dealloc]',
                    '-[MyString description]',
                    '-[MyString descriptionPauses]',     # synthesized property
                    # synthesized property
                    '-[MyString setDescriptionPauses:]',
                    'Test_Selector',
                    'Test_NSString',
                    'Test_MyString',
                    'Test_NSArray',
                    'main'
                    ]

    @add_test_categories(['pyapi'])
    def test_with_python_api(self):
        """Test symbol table access with Python APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        #
        # Exercise Python APIs to access the symbol table entries.
        #

        # Create the filespec by which to locate our a.out module.
        filespec = lldb.SBFileSpec(exe, False)

        module = target.FindModule(filespec)
        self.assertTrue(module, VALID_MODULE)

