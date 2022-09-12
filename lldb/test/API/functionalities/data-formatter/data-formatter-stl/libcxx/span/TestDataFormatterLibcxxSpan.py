"""
Test lldb data formatter subsystem for std::span
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

class LibcxxSpanDataFormatterTestCase(TestBase):

    def findVariable(self, name):
        var = self.frame().FindVariable(name)
        self.assertTrue(var.IsValid())
        return var

    def check_size(self, var_name, size):
        var = self.findVariable(var_name)
        self.assertEqual(var.GetNumChildren(), size)

    def check_numbers(self, var_name):
        """Helper to check that data formatter sees contents of std::span correctly"""

        expectedSize = 5
        self.check_size(var_name, expectedSize)

        self.expect_expr(
                var_name, result_type=f'std::span<int, {expectedSize}>',
                result_summary=f'size={expectedSize}',
                result_children=[
                    ValueCheck(name='[0]', value='1'),
                    ValueCheck(name='[1]', value='12'),
                    ValueCheck(name='[2]', value='123'),
                    ValueCheck(name='[3]', value='1234'),
                    ValueCheck(name='[4]', value='12345')
        ])

        # check access-by-index
        self.expect_var_path(f'{var_name}[0]', type='int', value='1')
        self.expect_var_path(f'{var_name}[1]', type='int', value='12')
        self.expect_var_path(f'{var_name}[2]', type='int', value='123')
        self.expect_var_path(f'{var_name}[3]', type='int', value='1234')
        self.expect_var_path(f'{var_name}[4]', type='int', value='12345')

    @add_test_categories(['libc++'])
    @skipIf(compiler='clang', compiler_version=['<', '11.0'])
    @skipIf(debug_info='gmodules', bugnumber="rdar://99758046") # Crashes Clang while compiling module. 
    def test_with_run_command(self):
        """Test that std::span variables are formatted correctly when printed."""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, 'break here', lldb.SBFileSpec('main.cpp', False))

        lldbutil.continue_to_breakpoint(process, bkpt)

        # std::span of std::array with extents known at compile-time
        self.check_numbers('numbers_span')

        # check access to synthetic children for static spans
        self.runCmd('type summary add --summary-string "item 0 is ${var[0]}" -x "std::span<" span')
        self.expect_expr('numbers_span',
                         result_type='std::span<int, 5>',
                         result_summary='item 0 is 1')

        self.runCmd('type summary add --summary-string "item 0 is ${svar[0]}" -x "std::span<" span')
        self.expect_expr('numbers_span',
                         result_type='std::span<int, 5>',
                         result_summary='item 0 is 1')

        self.runCmd('type summary delete span')

        # New span with strings
        lldbutil.continue_to_breakpoint(process, bkpt)

        expectedStringSpanChildren = [ ValueCheck(name='[0]', summary='"smart"'),
                                       ValueCheck(name='[1]', summary='"!!!"') ]

        self.expect_var_path('strings_span',
                             summary='size=2',
                             children=expectedStringSpanChildren)

        # check access to synthetic children for dynamic spans
        self.runCmd('type summary add --summary-string "item 0 is ${var[0]}" dynamic_string_span')
        self.expect_var_path('strings_span', summary='item 0 is "smart"')

        self.runCmd('type summary add --summary-string "item 0 is ${svar[0]}" dynamic_string_span')
        self.expect_var_path('strings_span', summary='item 0 is "smart"')

        self.runCmd('type summary delete dynamic_string_span')

        # test summaries based on synthetic children
        self.runCmd(
                'type summary add --summary-string "span has ${svar%#} items" -e dynamic_string_span')

        self.expect_var_path('strings_span', summary='span has 2 items')

        self.expect_var_path('strings_span',
                             summary='span has 2 items',
                             children=expectedStringSpanChildren)

        # check access-by-index
        self.expect_var_path('strings_span[0]', summary='"smart"');
        self.expect_var_path('strings_span[1]', summary='"!!!"');

        # Newly inserted value not visible to span
        lldbutil.continue_to_breakpoint(process, bkpt)

        self.expect_expr('strings_span',
                         result_summary='span has 2 items',
                         result_children=expectedStringSpanChildren)

        self.runCmd('type summary delete dynamic_string_span')

        lldbutil.continue_to_breakpoint(process, bkpt)

        # Empty spans
        self.expect_expr('static_zero_span',
                         result_type='std::span<int, 0>',
                         result_summary='size=0')
        self.check_size('static_zero_span', 0)

        self.expect_expr('dynamic_zero_span',
                         result_summary='size=0')
        self.check_size('dynamic_zero_span', 0)

        # Nested spans
        self.expect_expr('nested',
                         result_summary='size=2',
                         result_children=[
                             ValueCheck(name='[0]', summary='size=2',
                                        children=expectedStringSpanChildren),
                             ValueCheck(name='[1]', summary='size=2',
                                        children=expectedStringSpanChildren)
        ])
        self.check_size('nested', 2)

    @add_test_categories(['libc++'])
    @skipIf(compiler='clang', compiler_version=['<', '11.0'])
    @skipIf(debug_info='gmodules', bugnumber="rdar://99758046") # Crashes Clang while compiling module. 
    def test_ref_and_ptr(self):
        """Test that std::span is correctly formatted when passed by ref and ptr"""
        self.build()
        (self.target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, 'Stop here to check by ref', lldb.SBFileSpec('main.cpp', False))

        # The reference should display the same was as the value did
        self.check_numbers('ref')

        # The pointer should just show the right number of elements:

        ptrAddr = self.findVariable('ptr').GetValue()
        self.expect_expr('ptr', result_type='std::span<int, 5> *',
                         result_summary=f'{ptrAddr} size=5')
