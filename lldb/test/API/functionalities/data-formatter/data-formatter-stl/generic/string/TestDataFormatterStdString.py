# coding=utf8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdStringDataFormatterTestCase(TestBase):
    TEST_WITH_PDB_DEBUG_INFO = True

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.main_spec = lldb.SBFileSpec("main.cpp")
        self.namespace = "std"

    def _makeStringName(self, typedef: str, char_type: str, allocator=None):
        if allocator is None:
            allocator = self.namespace + "::allocator"

        if self.getDebugInfo() == "pdb":
            return f"{self.namespace}::basic_string<{char_type}, std::char_traits<{char_type}>, {allocator}<{char_type}>>"

        if typedef.startswith("::"):
            return self.namespace + typedef
        return typedef

    def do_test(self):
        """Test that that file and class static variables display correctly."""
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", self.main_spec
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

        string_name = self._makeStringName("::string", "char")
        wstring_name = self._makeStringName("::wstring", "wchar_t")
        custom_string_name = self._makeStringName(
            "CustomString", "char", allocator="CustomAlloc"
        )
        custom_wstring_name = self._makeStringName(
            "CustomWString", "wchar_t", allocator="CustomAlloc"
        )

        # Check 'S' pre-assignment.
        self.expect("frame variable S", substrs=[f'({wstring_name}) S = L"!!!!"'])

        thread.StepOver()

        TheVeryLongOne = frame.FindVariable("TheVeryLongOne")
        summaryOptions = lldb.SBTypeSummaryOptions()
        summaryOptions.SetCapping(lldb.eTypeSummaryCapped)
        cappedSummaryStream = lldb.SBStream()
        TheVeryLongOne.GetSummary(cappedSummaryStream, summaryOptions)
        cappedSummary = cappedSummaryStream.GetData()
        self.assertLessEqual(
            cappedSummary.find("someText"), 0, "cappedSummary includes the full string"
        )

        self.expect_expr(
            "s", result_type=wstring_name, result_summary='L"hello world! מזל טוב!"'
        )

        self.expect_expr("q", result_type=string_name, result_summary='"hello world"')

        self.expect_expr(
            "Q",
            result_type=string_name,
            result_summary='"quite a long std::strin with lots of info inside it"',
        )

        self.expect(
            "frame variable",
            substrs=[
                f'({wstring_name}) wempty = L""',
                f'({wstring_name}) s = L"hello world! מזל טוב!"',
                f'({wstring_name}) S = L"!!!!!"',
                "(const wchar_t *) mazeltov = 0x",
                'L"מזל טוב"',
                f'({string_name}) empty = ""',
                f'({string_name}) q = "hello world"',
                f'({string_name}) Q = "quite a long std::strin with lots of info inside it"',
                f"({string_name} *) null_str = nullptr",
                f'({custom_string_name}) custom_str = "hello!"',
                f'({custom_wstring_name}) custom_wstr = L"hello!"',
            ],
        )

        # Test references and pointers to std::string.
        var_rq = frame.FindVariable("rq")
        var_rQ = frame.FindVariable("rQ")
        var_pq = frame.FindVariable("pq")
        var_pQ = frame.FindVariable("pQ")

        self.assertEqual(var_rq.GetSummary(), '"hello world"', "rq summary wrong")
        self.assertEqual(
            var_rQ.GetSummary(),
            '"quite a long std::strin with lots of info inside it"',
            "rQ summary wrong",
        )
        self.assertEqual(var_pq.GetSummary(), '"hello world"', "pq summary wrong")
        self.assertEqual(
            var_pQ.GetSummary(),
            '"quite a long std::strin with lots of info inside it"',
            "pQ summary wrong",
        )

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

    @expectedFailureAll(
        bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android"
    )
    # Inline namespace is randomly ignored as Clang due to broken lookup inside
    # the std namespace.
    @expectedFailureAll(debug_info="gmodules")
    @add_test_categories(["libstdcxx"])
    def test_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test()

    @add_test_categories(["msvcstl"])
    def test_msvc(self):
        self.build()
        self.do_test()

    def do_test_multibyte(self):
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", self.main_spec
        )

        u16string_name = self._makeStringName("::u16string", "char16_t")
        u32string_name = self._makeStringName("::u32string", "char32_t")
        custom_u16string_name = self._makeStringName(
            "CustomStringU16", "char16_t", allocator="CustomAlloc"
        )
        custom_u32string_name = self._makeStringName(
            "CustomStringU32", "char32_t", allocator="CustomAlloc"
        )

        self.expect(
            "frame variable",
            substrs=[
                f'({u16string_name}) u16_string = u"ß水氶"',
                f'({u16string_name}) u16_empty = u""',
                f'({u32string_name}) u32_string = U"🍄🍅🍆🍌"',
                f'({u32string_name}) u32_empty = U""',
                f'({custom_u16string_name}) custom_u16 = u"ß水氶"',
                f'({custom_u16string_name}) custom_u16_empty = u""',
                f'({custom_u32string_name}) custom_u32 = U"🍄🍅🍆🍌"',
                f'({custom_u32string_name}) custom_u32_empty = U""',
            ],
        )

    @add_test_categories(["libc++"])
    def test_multibyte_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_multibyte()

    @expectedFailureAll(
        bugnumber="libstdc++ formatters don't support UTF-16/UTF-32 strings yet."
    )
    @add_test_categories(["libstdcxx"])
    def test_multibyte_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_multibyte()

    @add_test_categories(["msvcstl"])
    def test_multibyte_msvc(self):
        self.build()
        self.do_test_multibyte()

    def do_test_uncapped_summary(self):
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", self.main_spec
        )

        TheVeryLongOne = thread.frames[0].FindVariable("TheVeryLongOne")
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

    @add_test_categories(["libc++"])
    def test_uncapped_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_uncapped_summary()

    @expectedFailureAll(
        bugnumber="libstdc++ std::string summary provider doesn't obey summary options."
    )
    @add_test_categories(["libstdcxx"])
    def test_uncapped_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_uncapped_summary()

    @add_test_categories(["msvcstl"])
    def test_uncapped_msvc(self):
        self.build()
        self.do_test_uncapped_summary()

    def do_test_summary_unavailable(self):
        """
        Make sure that if the string is not readable, we give an error.
        """
        (_, _, thread, _) = lldbutil.run_to_source_breakpoint(
            self, "Break here to look at bad string", self.main_spec
        )

        var = thread.frames[0].FindVariable("in_str")
        self.assertTrue(var.GetError().Success(), "Found variable")
        summary = var.GetSummary()
        self.assertEqual(summary, "Summary Unavailable", "No summary for bad value")

    @add_test_categories(["libc++"])
    def test_unavailable_summary_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_summary_unavailable()

    @add_test_categories(["libstdcxx"])
    def test_unavailable_summary_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_summary_unavailable()

    @expectedFailureAll(
        bugnumber="MSVC std::string summary provider doesn't output a user-friendly message for invalid strings."
    )
    @add_test_categories(["msvcstl"])
    def test_unavailable_summary_msvc(self):
        self.build()
        self.do_test_summary_unavailable()

    def do_test_overwritten(self):
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", self.main_spec
        )

        self.expect_var_path("overwritten_zero", summary='"abc"')

    @add_test_categories(["libc++"])
    def test_overwritten_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_overwritten()

    @expectedFailureAll(
        bugnumber="libstdc++ format for non-null terminated std::string currently diverges from MSVC and libc++ formatter."
    )
    @add_test_categories(["libstdcxx"])
    def test_overwritten_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_overwritten()

    @add_test_categories(["msvcstl"])
    def test_overwritten_msvc(self):
        self.build()
        self.do_test_overwritten()

    def do_test_embedded_null(self):
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", self.main_spec
        )

        ns = self.namespace

        self.expect(
            "frame variable",
            substrs=[
                f'({self._makeStringName("::string", "char")}) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"',
                f'({self._makeStringName("::wstring", "wchar_t")}) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"',
            ],
        )

    @add_test_categories(["libc++"])
    def test_embedded_null_libcxx(self):
        self.build(dictionary={"USE_LIBCPP": 1})
        self.do_test_embedded_null()

    @expectedFailureAll(
        bugnumber="libstdc++ formatters incorrectly format std::string with embedded '\0' characters."
    )
    @add_test_categories(["libstdcxx"])
    def test_embedded_null_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_embedded_null()

    @add_test_categories(["msvcstl"])
    def test_embedded_null_msvc(self):
        self.build()
        self.do_test_embedded_null()
