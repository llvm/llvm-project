import re

from lldbsuite.test.lldbtest import TestBase
from lldbsuite.test.decorators import *


def expectedFailureDwarf(bugnumber=None):
    return expectedFailureAll(bugnumber, debug_info="dwarf")


class TestDecoratorsNoDebugInfoClass(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    @expectedFailureAll(debug_info="dwarf")
    def test_decorator_xfail(self):
        """Test that specifying a debug info category works for a NO_DEBUG_INFO_TESTCASE"""

    @expectedFailureDwarf
    def test_decorator_xfail_bare_decorator(self):
        """Same as test_decorator_xfail, but with a custom decorator w/ a bare syntax"""

    @expectedFailureDwarf()
    def test_decorator_xfail_decorator_empty_args(self):
        """Same as test_decorator_xfail, but with a custom decorator w/ no args"""

    @add_test_categories(["dwarf"])
    def test_add_test_categories(self):
        # Note: the "dwarf" test category is ignored, because we don't generate any debug info test variants
        self.assertIsNone(self.getDebugInfo())

    @expectedFailureAll
    def test_xfail_regexp(self):
        """Test that expectedFailureAll can be empty (but please just use expectedFailure)"""
        self.fail()

    @expectedFailureAll(compiler=re.compile(".*"))
    def test_xfail_regexp(self):
        """Test that xfail can take a regex as a matcher"""
        self.fail()

    @expectedFailureAll(compiler=no_match(re.compile(".*")))
    def test_xfail_no_match(self):
        """Test that xfail can take a no_match matcher"""
        pass

    @expectedFailureIf(condition=True)
    def test_xfail_condition_true(self):
        self.fail()

    @expectedFailureIf(condition=False)
    def test_xfail_condition_false(self):
        pass


class TestDecorators(TestBase):
    @expectedFailureAll(debug_info="dwarf")
    def test_decorator_xfail(self):
        """Test that expectedFailureAll fails for the debug_info variant"""
        if self.getDebugInfo() == "dwarf":
            self.fail()

    @skipIf(debug_info="dwarf")
    def test_decorator_skip(self):
        """Test that skipIf skips the debug_info variant"""
        self.assertNotEqual(self.getDebugInfo(), "dwarf")

    @expectedFailureDwarf
    def test_decorator_xfail2(self):
        """Same as test_decorator_xfail, but with a custom decorator w/ a bare syntax"""
        if self.getDebugInfo() == "dwarf":
            self.fail()

    @expectedFailureDwarf()
    def test_decorator_xfail3(self):
        """Same as test_decorator_xfail, but with a custom decorator w/ no args"""
        if self.getDebugInfo() == "dwarf":
            self.fail()

    @add_test_categories(["dwarf"])
    def test_add_test_categories(self):
        """Test that add_test_categories limits the kinds of debug info test variants"""
        self.assertEqual(self.getDebugInfo(), "dwarf")

    @expectedFailureAll(compiler="fake", debug_info="dwarf")
    def test_decorator_xfail_all(self):
        """Test that expectedFailureAll requires all conditions to match to be xfail"""

    @skipIf(compiler="fake", debug_info="dwarf")
    def test_decorator_skip2(self):
        """Test that expectedFailureAll fails for the debug_info variant"""
        # Note: the following assertion would fail, if this were not skipped:
        # self.assertNotEqual(self.getDebugInfo(), "dwarf")

    @expectedFailureAll
    def test_xfail_regexp(self):
        """Test that xfail can be empty"""
        self.fail()

    @expectedFailureAll(compiler=re.compile(".*"))
    def test_xfail_regexp(self):
        """Test that xfail can take a regex as a matcher"""
        self.fail()

    @expectedFailureAll(compiler=no_match(re.compile(".*")))
    def test_xfail_no_match(self):
        """Test that xfail can take a no_match matcher"""
        pass

    @expectedFailureIf(condition=True)
    def test_xfail_condition_true(self):
        self.fail()

    @expectedFailureIf(condition=False)
    def test_xfail_condition_false(self):
        pass
