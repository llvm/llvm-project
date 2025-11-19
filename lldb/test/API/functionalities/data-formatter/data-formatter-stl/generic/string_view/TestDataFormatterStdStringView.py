# coding=utf8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdStringViewDataFormatterTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line1 = line_number("main.cpp", "// Set break point at this line.")
        self.line2 = line_number(
            "main.cpp", "// Break here to look at bad string view."
        )

    def _makeStringName(self, typedef: str, char_type: str):
        if self.getDebugInfo() == "pdb":
            return f"std::basic_string_view<{char_type}, std::char_traits<{char_type}>>"

        return typedef

    def do_test(self):
        """Test that that file and class static variables display correctly."""
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line1, num_expected_locations=-1
        )
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line2, num_expected_locations=-1
        )

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        string_view_name = self._makeStringName("std::string_view", "char")
        wstring_view_name = self._makeStringName("std::wstring_view", "wchar_t")
        u16string_view_name = self._makeStringName("std::u16string_view", "char16_t")
        u32string_view_name = self._makeStringName("std::u32string_view", "char32_t")
        string_name = (
            "std::basic_string<char, std::char_traits<char>, std::allocator<char>>"
            if self.getDebugInfo() == "pdb"
            else "std::string"
        )

        self.expect_var_path("wempty", type=wstring_view_name, summary='L""')
        self.expect_var_path(
            "s", type=wstring_view_name, summary='L"hello world! מזל טוב!"'
        )
        self.expect_var_path("S", type=wstring_view_name, summary='L"!!!!"')
        self.expect_var_path("empty", type=string_view_name, summary='""')
        self.expect_var_path("q_source", type=string_name, summary='"hello world"')
        self.expect_var_path("q", type=string_view_name, summary='"hello world"')
        self.expect_var_path(
            "Q",
            type=string_view_name,
            summary='"quite a long std::strin with lots of info inside it"',
        )
        self.expect_var_path(
            "IHaveEmbeddedZeros", type=string_view_name, summary='"a\\0b\\0c\\0d"'
        )
        self.expect_var_path(
            "IHaveEmbeddedZerosToo",
            type=wstring_view_name,
            summary='L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"',
        )
        self.expect_var_path("u16_string", type=u16string_view_name, summary='u"ß水氶"')
        self.expect_var_path("u16_empty", type=u16string_view_name, summary='u""')
        self.expect_var_path("u32_string", type=u32string_view_name, summary='U"🍄🍅🍆🍌"')
        self.expect_var_path("u32_empty", type=u32string_view_name, summary='U""')

        # GetSummary returns None so can't be checked by expect_var_path, so we
        # use the str representation instead
        null_obj = self.frame().GetValueForVariablePath("null_str")
        self.assertEqual(null_obj.GetSummary(), "Summary Unavailable")
        self.assertEqual(str(null_obj), f"({string_view_name} *) null_str = nullptr")

        self.runCmd("n")

        TheVeryLongOne = self.frame().FindVariable("TheVeryLongOne")
        summaryOptions = lldb.SBTypeSummaryOptions()
        summaryOptions.SetCapping(lldb.eTypeSummaryUncapped)
        uncappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(uncappedSummaryStream, summaryOptions)
        uncappedSummary = uncappedSummaryStream.GetData()
        self.assertGreater(
            uncappedSummary.find("someText"),
            0,
            "uncappedSummary does not include the full string",
        )
        summaryOptions.SetCapping(lldb.eTypeSummaryCapped)
        cappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(cappedSummaryStream, summaryOptions)
        cappedSummary = cappedSummaryStream.GetData()
        self.assertLessEqual(
            cappedSummary.find("someText"), 0, "cappedSummary includes the full string"
        )

        self.expect_expr(
            "s",
            result_type=wstring_view_name,
            result_summary='L"hello world! מזל טוב!"',
        )

        self.expect_var_path("wempty", type=wstring_view_name, summary='L""')
        self.expect_var_path(
            "s", type=wstring_view_name, summary='L"hello world! מזל טוב!"'
        )
        self.expect_var_path("S", type=wstring_view_name, summary='L"!!!!"')
        self.expect_var_path("empty", type=string_view_name, summary='""')
        self.expect_var_path("q_source", type=string_name, summary='"Hello world"')
        self.expect_var_path("q", type=string_view_name, summary='"Hello world"')
        self.expect_var_path(
            "Q",
            type=string_view_name,
            summary='"quite a long std::strin with lots of info inside it"',
        )
        self.expect_var_path(
            "IHaveEmbeddedZeros", type=string_view_name, summary='"a\\0b\\0c\\0d"'
        )
        self.expect_var_path(
            "IHaveEmbeddedZerosToo",
            type=wstring_view_name,
            summary='L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"',
        )
        self.expect_var_path("u16_string", type=u16string_view_name, summary='u"ß水氶"')
        self.expect_var_path("u16_empty", type=u16string_view_name, summary='u""')
        self.expect_var_path("u32_string", type=u32string_view_name, summary='U"🍄🍅🍆🍌"')
        self.expect_var_path("u32_empty", type=u32string_view_name, summary='U""')

        self.runCmd("cont")
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

        broken_obj = self.frame().GetValueForVariablePath("in_str_view")
        self.assertEqual(broken_obj.GetSummary(), "Summary Unavailable")

    @expectedFailureAll(
        bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android"
    )
    # Inline namespace is randomly ignored as Clang due to broken lookup inside
    # the std namespace.
    @expectedFailureAll(debug_info="gmodules")
    @add_test_categories(["libc++"])
    def test_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvcstl(self):
        self.build()
        self.do_test()
