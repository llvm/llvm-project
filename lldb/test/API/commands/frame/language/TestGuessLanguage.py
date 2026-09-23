"""
Test the SB API SBFrame::GuessLanguage.
"""


import lldb
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class TestFrameGuessLanguage(TestBase):
    # If your test case doesn't stress debug info, then
    # set this to true.  That way it won't be run once for
    # each debug info format.
    NO_DEBUG_INFO_TESTCASE = True

    @skipIf(compiler="clang", compiler_version=["<", "10.0"])
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37658")
    def test_guess_language(self):
        """Test GuessLanguage for C and C++."""
        self.build()
        self.do_test()

    def check_language(self, thread, frame_no, test_lang):
        frame = thread.frames[frame_no]
        self.assertTrue(frame.IsValid(), "Frame %d was not valid." % (frame_no))
        lang = frame.GuessLanguage()
        self.assertEqual(lang, test_lang)

    def do_test(self):
        """Test GuessLanguage for C & C++."""
        _, _, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "Set breakpoint here", lldb.SBFileSpec("somefunc.c")
        )

        c_frame_language = lldb.eLanguageTypeC99
        cxx_frame_language = lldb.eLanguageTypeC_plus_plus_11
        # gcc emits DW_LANG_C89 even if -std=c99 was specified
        if "gcc" in self.getCompiler():
            c_frame_language = lldb.eLanguageTypeC89
            cxx_frame_language = lldb.eLanguageTypeC_plus_plus

        self.check_language(thread, 0, c_frame_language)
        self.check_language(thread, 1, cxx_frame_language)
        self.check_language(thread, 2, lldb.eLanguageTypeC_plus_plus)
