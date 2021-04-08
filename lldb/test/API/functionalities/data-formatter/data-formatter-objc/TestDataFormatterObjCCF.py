# encoding: utf-8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil

from ObjCDataFormatterTestCase import ObjCDataFormatterTestCase


class ObjCDataFormatterCF(ObjCDataFormatterTestCase):

    def test_coreframeworks_and_run_command(self):
        """Test formatters for Core OSX frameworks."""
        self.build()
        self.target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, '// Set break point at this line.',
            lldb.SBFileSpec('main.m', False))

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        # check formatters for common Objective-C types
        expect_strings = [
            '(CFGregorianUnits) cf_greg_units = 1 years, 3 months, 5 days, 12 hours, 5 minutes 7 seconds',
            '(CFGregorianDate) cf_greg_date = @"4/10/1985 18:0:0"',
            '(CFRange) cf_range = location=4 length=4',
            '(NSPoint) ns_point = (x = 4, y = 4)',
            '(NSRange) ns_range = location=4, length=4',
            '(NSRect) ns_rect = (origin = (x = 1, y = 1), size = (width = 5, height = 5))',
            '(NSRectArray) ns_rect_arr = ((x = 1, y = 1), (width = 5, height = 5)), ...',
            '(NSSize) ns_size = (width = 5, height = 7)',
            '(CGSize) cg_size = (width = 1, height = 6)',
            '(CGPoint) cg_point = (x = 2, y = 7)',
            '(CGRect) cg_rect = (origin = (x = 1, y = 2), size = (width = 7, height = 7))',
            '(Rect) rect = (t=4, l=8, b=4, r=7)',
            '(Rect *) rect_ptr = (t=4, l=8, b=4, r=7)',
            '(Point) point = (v=7, h=12)', '(Point *) point_ptr = (v=7, h=12)',
            '(SEL) foo_selector = "foo_selector_impl"'
        ]
        self.expect("frame variable", substrs=expect_strings)

        if self.getArchitecture() in ['i386', 'x86_64']:
            extra_string = [
                '(RGBColor) rgb_color = red=3 green=56 blue=35',
                '(RGBColor *) rgb_color_ptr = red=3 green=56 blue=35',
                '(HIPoint) hi_point = (x=7, y=12)',
                '(HIRect) hi_rect = origin=(x = 3, y = 5) size=(width = 4, height = 6)',
            ]
            self.expect("frame variable", substrs=extra_string)

        # The original tests left out testing the NSNumber values, so do that here.
        # This set is all pointers, with summaries, so we only check the summary.
        var_list_pointer = [
            ['NSNumber *', 'num1',    '(int)5'],
            ['NSNumber *', 'num2',    '(float)3.140000'],
            ['NSNumber *', 'num3',    '(double)3.14'],
            ['NSNumber *', 'num4',    '(int128_t)18446744073709551614'],
            ['NSNumber *', 'num5',    '(char)65'],
            ['NSNumber *', 'num6',    '(long)255'],
            ['NSNumber *', 'num7',    '(long)2000000'],
            ['NSNumber *', 'num8_Y',  'YES'],
            ['NSNumber *', 'num8_N',  'NO'],
            ['NSNumber *', 'num9',    '(short)-31616'],
            ['NSNumber *', 'num_at1', '(int)12'],
            ['NSNumber *', 'num_at2', '(int)-12'],
            ['NSNumber *', 'num_at3', '(double)12.5'],
            ['NSNumber *', 'num_at4', '(double)-12.5'],
            ['NSDecimalNumber *', 'decimal_number', '123456 x 10^-10'],
            ['NSDecimalNumber *', 'decimal_number_neg', '-123456 x 10^10']
        ]
        for type, var_path, summary in var_list_pointer:
            self.expect_var_path(var_path, summary, None, type)

            
