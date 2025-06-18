# coding=utf8
"""
Test std::*string summaries with MSVC's STL.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class MsvcStlStringDataFormatterTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @add_test_categories(["msvcstl"])
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()

        main_spec = lldb.SBFileSpec("main.cpp")
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", main_spec
        )
        frame = thread.frames[0]

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd("type format clear", check=False)
            self.runCmd("type summary clear", check=False)
            self.runCmd("type filter clear", check=False)
            self.runCmd("type synth clear", check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        self.expect(
            "frame variable",
            substrs=[
                '(std::wstring) wempty = L""',
                '(std::wstring) s = L"hello world! מזל טוב!"',
                '(std::wstring) S = L"!!!!"',
                "(const wchar_t *) mazeltov = 0x",
                'L"מזל טוב"',
                '(std::string) empty = ""',
                '(std::string) q = "hello world"',
                '(std::string) Q = "quite a long std::strin with lots of info inside it"',
                '(std::string) overwritten_zero = "abc"',
                '(std::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"',
                '(std::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"',
                '(std::u16string) u16_string = u"ß水氶"',
                '(std::u16string) u16_empty = u""',
                '(std::u32string) u32_string = U"🍄🍅🍆🍌"',
                '(std::u32string) u32_empty = U""',
                "(std::string *) null_str = nullptr",
            ],
        )

        thread.StepOver()

        TheVeryLongOne = frame.FindVariable("TheVeryLongOne")
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
            "s", result_type="std::wstring", result_summary='L"hello world! מזל טוב!"'
        )

        self.expect_expr("q", result_type="std::string", result_summary='"hello world"')

        self.expect_expr(
            "Q",
            result_type="std::string",
            result_summary='"quite a long std::strin with lots of info inside it"',
        )

        self.expect(
            "frame variable",
            substrs=[
                '(std::wstring) S = L"!!!!!"',
                "(const wchar_t *) mazeltov = 0x",
                'L"מזל טוב"',
                '(std::string) q = "hello world"',
                '(std::string) Q = "quite a long std::strin with lots of info inside it"',
                '(std::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"',
                '(std::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"',
                '(std::u16string) u16_string = u"ß水氶"',
                '(std::u32string) u32_string = U"🍄🍅🍆🍌"',
                '(std::u32string) u32_empty = U""',
                "(std::string *) null_str = nullptr",
            ],
        )

        # Finally, make sure that if the string is not readable, we give an error:
        bkpt_2 = target.BreakpointCreateBySourceRegex(
            "Break here to look at bad string", main_spec
        )
        self.assertEqual(bkpt_2.GetNumLocations(), 1, "Got one location")
        threads = lldbutil.continue_to_breakpoint(process, bkpt_2)
        self.assertEqual(len(threads), 1, "Stopped at second breakpoint")
        frame = threads[0].frames[0]
        var = frame.FindVariable("in_str")
        self.assertTrue(var.GetError().Success(), "Made variable")
        self.assertIsNone(var.GetSummary())
