"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class LanguageCategoryUpdatesTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.cpp', '// break here')

    def test_with_run_command(self):
        """Test that LLDB correctly cleans caches when language categories change."""
        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            if hasattr(
                    self,
                    'type_category') and hasattr(
                    self,
                    'type_specifier'):
                self.type_category.DeleteTypeSummary(self.type_specifier)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        self.expect(
            "frame variable",
            substrs=[
                '(S)',
                'object',
                '123',
                '456'],
            matching=True)

        self.type_category = self.dbg.GetCategory(
            lldb.eLanguageTypeC_plus_plus)
        type_summary = lldb.SBTypeSummary.CreateWithSummaryString(
            "this is an object of type S")
        self.type_specifier = lldb.SBTypeNameSpecifier('S')
        self.type_category.AddTypeSummary(self.type_specifier, type_summary)

        self.expect(
            "frame variable",
            substrs=['this is an object of type S'],
            matching=True)

        self.type_category.DeleteTypeSummary(self.type_specifier)
        self.expect(
            "frame variable",
            substrs=['this is an object of type S'],
            matching=False)
        self.expect(
            "frame variable",
            substrs=[
                '(S)',
                'object',
                '123',
                '456'],
            matching=True)
