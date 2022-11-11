"""
Test the session history command
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class SessionHistoryTestCase(TestBase):

    @no_debug_info_test
    def test_history(self):
        self.runCmd('session history --clear', inHistory=False)
        self.runCmd('breakpoint list', check=False, inHistory=True)  # 0
        self.runCmd('register read', check=False, inHistory=True)  # 1
        self.runCmd('apropos hello', check=False, inHistory=True)  # 2
        self.runCmd('memory write', check=False, inHistory=True)  # 3
        self.runCmd('log list', check=False, inHistory=True)  # 4
        self.runCmd('disassemble', check=False, inHistory=True)  # 5
        self.runCmd('expression 1', check=False, inHistory=True)  # 6
        self.runCmd(
            'type summary list -w default',
            check=False,
            inHistory=True)  # 7
        self.runCmd('version', check=False, inHistory=True)  # 8
        self.runCmd('frame select 1', check=False, inHistory=True)  # 9

        self.expect(
            "session history -s 3 -c 3",
            inHistory=True,
            substrs=[
                '3: memory write',
                '4: log list',
                '5: disassemble'])

        self.expect("session history -s 3 -e 3", inHistory=True,
                    substrs=['3: memory write'])

        self.expect(
            "session history -s 6 -e 7",
            inHistory=True,
            substrs=[
                '6: expression 1',
                '7: type summary list -w default'])

        self.expect("session history -c 2", inHistory=True,
                    substrs=['0: breakpoint list', '1: register read'])

        self.expect("session history -e 3 -c 1", inHistory=True,
                    substrs=['3: memory write'])

        self.expect(
            "session history -e 2",
            inHistory=True,
            substrs=[
                '0: breakpoint list',
                '1: register read',
                '2: apropos hello'])

        self.expect(
            "session history -s 12",
            inHistory=True,
            substrs=[
                '12: session history -s 6 -e 7',
                '13: session history -c 2',
                '14: session history -e 3 -c 1',
                '15: session history -e 2',
                '16: session history -s 12'])

        self.expect(
            "session history -s end -c 3",
            inHistory=True,
            substrs=[
                '15: session history -e 2',
                '16: session history -s 12',
                '17: session history -s end -c 3'])

        self.expect(
            "session history -s end -e 15",
            inHistory=True,
            substrs=[
                '15: session history -e 2',
                '16: session history -s 12',
                '17: session history -s end -c 3',
                'session history -s end -e 15'])

        self.expect("session history -s 5 -c 1", inHistory=True,
                    substrs=['5: disassemble'])

        self.expect("session history -c 1 -s 5", inHistory=True,
                    substrs=['5: disassemble'])

        self.expect("session history -c 1 -e 3", inHistory=True,
                    substrs=['3: memory write'])

        self.expect(
            "session history -c 1 -e 3 -s 5",
            error=True,
            inHistory=True,
            substrs=['error: --count, --start-index and --end-index cannot be all specified in the same invocation'])
