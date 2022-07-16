import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class InvalidArgsExpressionTestCase(TestBase):

    @no_debug_info_test
    def test_invalid_lang(self):
        self.expect("expression -l foo --", error=True,
                    substrs=["error: unknown language type: 'foo' for expression"])

    @no_debug_info_test
    def test_invalid_all_thread(self):
        self.expect("expression -a foo --", error=True,
                    substrs=['error: invalid all-threads value setting: "foo"'])

    @no_debug_info_test
    def test_invalid_ignore_br(self):
        self.expect("expression -i foo --", error=True,
                    substrs=['error: could not convert "foo" to a boolean value.'])

    @no_debug_info_test
    def test_invalid_allow_jit(self):
        self.expect("expression -j foo --", error=True,
                    substrs=['error: could not convert "foo" to a boolean value.'])

    @no_debug_info_test
    def test_invalid_timeout(self):
        self.expect("expression -t foo --", error=True,
                    substrs=['error: invalid timeout setting "foo"'])

        self.expect("expression -t \"\" --", error=True,
                    substrs=['error: invalid timeout setting ""'])

    @no_debug_info_test
    def test_invalid_unwind(self):
        self.expect("expression -u foo --", error=True,
                    substrs=['error: could not convert "foo" to a boolean value.'])

    @no_debug_info_test
    def test_invalid_fixits(self):
        self.expect("expression -X foo --", error=True,
                    substrs=['error: could not convert "foo" to a boolean value.'])
