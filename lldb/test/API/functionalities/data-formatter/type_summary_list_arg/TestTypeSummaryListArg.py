"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TypeSummaryListArgumentTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_type_summary_list_with_arg(self):
        """Test that the 'type summary list' command handles command line arguments properly"""
        self.expect(
            'type summary list Foo',
            substrs=[
                'Category: default',
                'Category: system'])
        self.expect(
            'type summary list char',
            substrs=[
                'char ?(\*|\[\])',
                'char ?\[[0-9]+\]'])

        self.expect(
            'type summary list -w default',
            substrs=['system'],
            matching=False)
        self.expect(
            'type summary list -w system unsigned',
            substrs=[
                'default',
                '0-9'],
            matching=False)
        self.expect(
            'type summary list -w system char',
            substrs=[
                'char ?(\*|\[\])',
                'char ?\[[0-9]+\]'],
            matching=True)
