# coding=utf8
"""
Test that C++ supports wchar_t correctly.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class CxxWCharTTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def test(self):
        """Test that C++ supports wchar_t correctly."""
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here", lldb.SBFileSpec("main.cpp"))

        # Check that we correctly report templates on wchar_t
        self.expect("frame variable foo_y",
                    substrs=['(Foo<wchar_t>) foo_y = '])

        # Check that we correctly report templates on int
        self.expect("frame variable foo_x",
                    substrs=['(Foo<int>) foo_x = '])

        # Check that we correctly report wchar_t
        self.expect("frame variable foo_y.object",
                    substrs=['(wchar_t) foo_y.object = '])

        # Check that we correctly report int
        self.expect("frame variable foo_x.object",
                    substrs=['(int) foo_x.object = '])

        # Check that we can run expressions that return wchar_t
        self.expect("expression L'a'", substrs=['(wchar_t) $', "L'a'"])

        # Mazel Tov if this works!
        self.expect("frame variable mazeltov",
                    substrs=['(const wchar_t *) mazeltov = ', 'L"מזל טוב"'])

        self.expect(
            "frame variable ws_NULL",
            substrs=['(wchar_t *) ws_NULL = 0x0'])
        self.expect("frame variable ws_empty", substrs=[' L""'])

        self.expect("frame variable array", substrs=[
                    'L"Hey, I\'m a super wchar_t string'])
        self.expect("frame variable array", substrs=['[0]'], matching=False)

        self.expect('frame variable wchar_zero', substrs=["L'\\0'"])
        self.expect('expression wchar_zero', substrs=["L'\\0'"])
