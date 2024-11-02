# coding=utf8
"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class LibcxxStringDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.main_spec = lldb.SBFileSpec("main.cpp")
        self.namespace = 'std'

    @add_test_categories(["libc++"])
    @expectedFailureAll(bugnumber="llvm.org/pr36109", debug_info="gmodules", triple=".*-android")
    # Inline namespace is randomly ignored as Clang due to broken lookup inside
    # the std namespace.
    @expectedFailureAll(debug_info="gmodules")
    def test_with_run_command(self):
        """Test that that file and class static variables display correctly."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                                                            "Set break point at this line.",
                                                                            self.main_spec)
        frame = thread.frames[0]

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

        is_64_bit = self.process().GetAddressByteSize() == 8

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        ns = self.namespace
        self.expect(
            "frame variable",
            substrs=[
                '(%s::wstring) wempty = L""'%ns,
                '(%s::wstring) s = L"hello world! מזל טוב!"'%ns,
                '(%s::wstring) S = L"!!!!"'%ns,
                '(const wchar_t *) mazeltov = 0x',
                'L"מזל טוב"',
                '(%s::string) empty = ""'%ns,
                '(%s::string) q = "hello world"'%ns,
                '(%s::string) Q = "quite a long std::strin with lots of info inside it"'%ns,
                '(%s::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"'%ns,
                '(%s::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"'%ns,
                '(%s::u16string) u16_string = u"ß水氶"'%ns,
                # FIXME: This should have a 'u' prefix.
                '(%s::u16string) u16_empty = ""'%ns,
                '(%s::u32string) u32_string = U"🍄🍅🍆🍌"'%ns,
                # FIXME: This should have a 'U' prefix.
                '(%s::u32string) u32_empty = ""'%ns,
                '(%s::basic_string<unsigned char, %s::char_traits<unsigned char>, '
                '%s::allocator<unsigned char> >) uchar = "aaaaa"'%(ns,ns,ns),
                '(%s::string *) null_str = nullptr'%ns,
        ])

        thread.StepOver()

        TheVeryLongOne = frame.FindVariable("TheVeryLongOne")
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

        self.expect_expr("s", result_type=ns+"::wstring", result_summary='L"hello world! מזל טוב!"')

        self.expect(
            "frame variable",
            substrs=[
                '(%s::wstring) S = L"!!!!!"'%ns,
                '(const wchar_t *) mazeltov = 0x',
                'L"מזל טוב"',
                '(%s::string) q = "hello world"'%ns,
                '(%s::string) Q = "quite a long std::strin with lots of info inside it"'%ns,
                '(%s::string) IHaveEmbeddedZeros = "a\\0b\\0c\\0d"'%ns,
                '(%s::wstring) IHaveEmbeddedZerosToo = L"hello world!\\0てざ ル゜䋨ミ㠧槊 きゅへ狦穤襩 じゃ馩リョ 䤦監"'%ns,
                '(%s::u16string) u16_string = u"ß水氶"'%ns,
                '(%s::u32string) u32_string = U"🍄🍅🍆🍌"'%ns,
                '(%s::u32string) u32_empty = ""'%ns,
                '(%s::basic_string<unsigned char, %s::char_traits<unsigned char>, '
                '%s::allocator<unsigned char> >) uchar = "aaaaa"'%(ns,ns,ns),
                '(%s::string *) null_str = nullptr'%ns,
        ])

        # The test assumes that std::string is in its cap-size-data layout.
        is_alternate_layout = ('arm' in self.getArchitecture()) and self.platformIsDarwin()
        if is_64_bit and not is_alternate_layout:
            self.expect("frame variable garbage1", substrs=['garbage1 = Summary Unavailable'])
            self.expect("frame variable garbage2", substrs=[r'garbage2 = "\xfa\xfa\xfa\xfa"'])
            self.expect("frame variable garbage3", substrs=[r'garbage3 = "\xf0\xf0"'])
            self.expect("frame variable garbage4", substrs=['garbage4 = Summary Unavailable'])
            self.expect("frame variable garbage5", substrs=['garbage5 = Summary Unavailable'])

        # Finally, make sure that if the string is not readable, we give an error:
        bkpt_2 = target.BreakpointCreateBySourceRegex("Break here to look at bad string", self.main_spec)
        self.assertEqual(bkpt_2.GetNumLocations(), 1, "Got one location")
        threads = lldbutil.continue_to_breakpoint(process, bkpt_2)
        self.assertEqual(len(threads), 1, "Stopped at second breakpoint")
        frame = threads[0].frames[0]
        var = frame.FindVariable("in_str")
        self.assertTrue(var.GetError().Success(), "Made variable")
        summary = var.GetSummary()
        self.assertEqual(summary, "Summary Unavailable", "No summary for bad value")
        
