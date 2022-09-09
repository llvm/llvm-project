# coding=utf8
"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxStringViewDataFormatterTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line1 = line_number('main.cpp', '// Set break point at this line.')
        self.line2 = line_number('main.cpp', '// Break here to look at bad string view.' )

    @add_test_categories(["libc++"])
    @expectedFailureDarwin # FIXME: May need to force system libcxx here.
    @expectedFailureAll(bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android")
    # Inline namespace is randomly ignored as Clang due to broken lookup inside
    # the std namespace.
    @expectedFailureAll(debug_info="gmodules")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line1, num_expected_locations=-1)
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line2, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type filter clear', check=False)
            self.runCmd('type synth clear', check=False)
            self.runCmd(
                "settings set target.max-children-count 256",
                check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect_var_path('wempty',
                             type='std::wstring_view',
                             summary='L""')
        self.expect_var_path('s',
                             type='std::wstring_view',
                             summary='L"hello world! מזל טוב!"')
        self.expect_var_path('S',
                             type='std::wstring_view',
                             summary='L"!!!!"')
        self.expect_var_path('empty',
                             type='std::string_view',
                             summary='""')
        self.expect_var_path('q_source',
                             type='std::string',
                             summary='"hello world"')
        self.expect_var_path('q',
                             type='std::string_view',
                             summary='"hello world"')
        self.expect_var_path('Q',
                             type='std::string_view',
                             summary='"quite a long std::strin with lots of info inside it"')
        self.expect_var_path('IHaveEmbeddedZeros',
                             type='std::string_view',
                             summary='"a\\0b\\0c\\0d"')
        self.expect_var_path('IHaveEmbeddedZerosToo',
                             type='std::wstring_view',
                             summary='L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"')
        self.expect_var_path('u16_string',
                             type='std::u16string_view',
                             summary='u"ß水氶"')
        self.expect_var_path('u16_empty',
                             type='std::u16string_view',
                             summary='""')
        self.expect_var_path('u32_string',
                             type='std::u32string_view',
                             summary='U"🍄🍅🍆🍌"')
        self.expect_var_path('u32_empty',
                             type='std::u32string_view',
                             summary='""')
        self.expect_var_path('uchar_source',
                             type='std::basic_string<unsigned char, std::char_traits<unsigned char>, std::allocator<unsigned char> >',
                             summary='"aaaaaaaaaa"')
        self.expect_var_path('uchar',
                             type='std::basic_string_view<unsigned char, std::char_traits<unsigned char> >',
                             summary='"aaaaa"')
        self.expect_var_path('oops',
                             type='std::string_view',
                             summary='"Hellooo World\\n"')

        # GetSummary returns None so can't be checked by expect_var_path, so we
        # use the str representation instead
        null_obj = self.frame().GetValueForVariablePath('null_str')
        self.assertEqual(null_obj.GetSummary(), "Summary Unavailable")
        self.assertEqual(str(null_obj),
                         '(std::string_view *) null_str = nullptr');

        self.runCmd("n")

        TheVeryLongOne = self.frame().FindVariable("TheVeryLongOne")
        summaryOptions = lldb.SBTypeSummaryOptions()
        summaryOptions.SetCapping(lldb.eTypeSummaryUncapped)
        uncappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(uncappedSummaryStream, summaryOptions)
        uncappedSummary = uncappedSummaryStream.GetData()
        self.assertTrue(uncappedSummary.find("someText") > 0,
                        "uncappedSummary does not include the full string")
        summaryOptions.SetCapping(lldb.eTypeSummaryCapped)
        cappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(cappedSummaryStream, summaryOptions)
        cappedSummary = cappedSummaryStream.GetData()
        self.assertTrue(
            cappedSummary.find("someText") <= 0,
            "cappedSummary includes the full string")

        self.expect_expr("s", result_type="std::wstring_view", result_summary='L"hello world! מזל טוב!"')

        self.expect_var_path('wempty',
                             type='std::wstring_view',
                             summary='L""')
        self.expect_var_path('s',
                             type='std::wstring_view',
                             summary='L"hello world! מזל טוב!"')
        self.expect_var_path('S',
                             type='std::wstring_view',
                             summary='L"!!!!"')
        self.expect_var_path('empty',
                             type='std::string_view',
                             summary='""')
        self.expect_var_path('q_source',
                             type='std::string',
                             summary='"Hello world"')
        self.expect_var_path('q',
                             type='std::string_view',
                             summary='"Hello world"')
        self.expect_var_path('Q',
                             type='std::string_view',
                             summary='"quite a long std::strin with lots of info inside it"')
        self.expect_var_path('IHaveEmbeddedZeros',
                             type='std::string_view',
                             summary='"a\\0b\\0c\\0d"')
        self.expect_var_path('IHaveEmbeddedZerosToo',
                             type='std::wstring_view',
                             summary='L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"')
        self.expect_var_path('u16_string',
                             type='std::u16string_view',
                             summary='u"ß水氶"')
        self.expect_var_path('u16_empty',
                             type='std::u16string_view',
                             summary='""')
        self.expect_var_path('u32_string',
                             type='std::u32string_view',
                             summary='U"🍄🍅🍆🍌"')
        self.expect_var_path('u32_empty',
                             type='std::u32string_view',
                             summary='""')
        self.expect_var_path('uchar_source',
                             type='std::basic_string<unsigned char, std::char_traits<unsigned char>, std::allocator<unsigned char> >',
                             summary='"aaaaaaaaaa"')
        self.expect_var_path('uchar',
                             type='std::basic_string_view<unsigned char, std::char_traits<unsigned char> >',
                             summary='"aaaaa"')
 
        self.runCmd('cont')
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        broken_obj = self.frame().GetValueForVariablePath('in_str_view')
        self.assertEqual( broken_obj.GetSummary(), "Summary Unavailable" )
