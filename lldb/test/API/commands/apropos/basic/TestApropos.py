import lldb
from lldbsuite.test.lldbtest import *
from lldbsuite.test.decorators import *

class AproposTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @no_debug_info_test
    def test_apropos(self):
        self.expect("apropos", error=True,
                    substrs=[' must be called with exactly one argument'])
        self.expect("apropos a b", error=True,
                    substrs=[' must be called with exactly one argument'])
        self.expect("apropos ''", error=True,
                    substrs=['\'\' is not a valid search word'])

    @no_debug_info_test
    def test_apropos_variable(self):
        """Test that 'apropos variable' prints the fully qualified command name"""
        self.expect(
            'apropos variable',
            substrs=[
                'frame variable',
                'target variable',
                'watchpoint set variable'])
