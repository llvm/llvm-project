# coding=utf8
"""
Test lldb data formatter subsystem.
"""


import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class StdStringDataFormatterTestCase(TestBase):
    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.main_spec = lldb.SBFileSpec("main.cpp")
        self.namespace = "std"

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

        ns = self.namespace

        # Check 'S' pre-assignment.
        self.expect("frame variable S", substrs=['(%s::wstring) S = L"!!!!"' % ns])

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
            "s", result_type=ns + "::wstring", result_summary='L"hello world! מזל טוב!"'
        )

        self.expect_expr(
            "q", result_type=ns + "::string", result_summary='"hello world"'
        )

        self.expect_expr(
            "Q",
            result_type=ns + "::string",
            result_summary='"quite a long std::strin with lots of info inside it"',
        )

        self.expect(
            "frame variable",
            substrs=[
                '(%s::wstring) wempty = L""' % ns,
                '(%s::wstring) s = L"hello world! מזל טוב!"' % ns,
                '(%s::wstring) S = L"!!!!!"' % ns,
                "(const wchar_t *) mazeltov = 0x",
                'L"מזל טוב"',
                '(%s::string) empty = ""' % ns,
                '(%s::string) q = "hello world"' % ns,
                '(%s::string) Q = "quite a long std::strin with lots of info inside it"'
                % ns,
                "(%s::string *) null_str = nullptr" % ns,
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

    def do_test_multibyte(self):
        lldbutil.run_to_source_breakpoint(
            self, "Set break point at this line.", self.main_spec
        )

        ns = self.namespace

        self.expect(
            "frame variable",
            substrs=[
                '(%s::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"' % ns,
                '(%s::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"'
                % ns,
                '(%s::u16string) u16_string = u"ß水氶"' % ns,
                '(%s::u16string) u16_empty = u""' % ns,
                '(%s::u32string) u32_string = U"🍄🍅🍆🍌"' % ns,
                '(%s::u32string) u32_empty = U""' % ns,
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

    @expectedFailureAll(
        bugnumber="libstdc++ std::string summary provider doesn't output a user-friendly message for invalid strings."
    )
    @add_test_categories(["libstdcxx"])
    def test_unavailable_summary_libstdcxx(self):
        self.build(dictionary={"USE_LIBSTDCPP": 1})
        self.do_test_summary_unavailable()
