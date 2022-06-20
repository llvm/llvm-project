"""
Test lldb data formatter subsystem.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class NSDictionarySyntheticTestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break at.
        self.line = line_number('main.m', '// Set break point at this line.')

    @skipUnlessDarwin
    def test_rdar11988289_with_run_command(self):
        """Test that NSDictionary reports its synthetic children properly."""
        self.build()
        self.run_tests()

    @skipUnlessDarwin
    def test_rdar11988289_with_run_command_no_const(self):
        """Test that NSDictionary reports its synthetic children properly."""
        disable_constant_classes = {
            'CC':
            'xcrun clang',  # FIXME: Remove when flags are available upstream.
            'CFLAGS_EXTRAS':
            '-fno-constant-nsnumber-literals ' +
            '-fno-constant-nsarray-literals ' +
            '-fno-constant-nsdictionary-literals'
        }
        self.build(dictionary=disable_constant_classes)
        self.run_tests()

    def run_tests(self):
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped',
                             'stop reason = breakpoint'])

        # This is the function to remove the custom formats in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type format clear', check=False)
            self.runCmd('type summary clear', check=False)
            self.runCmd('type synth clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        # Now check that we are displaying Cocoa classes correctly
        self.expect('frame variable dictionary',
                    substrs=['3 key/value pairs'])
        self.expect('frame variable mutabledict',
                    substrs=['4 key/value pairs'])
        self.expect(
            'frame variable dictionary --ptr-depth 1',
            substrs=[
                '3 key/value pairs',
                '[0] = ',
                'key = 0x',
                'value = 0x',
                '[1] = ',
                '[2] = '])
        self.expect(
            'frame variable mutabledict --ptr-depth 1',
            substrs=[
                '4 key/value pairs',
                '[0] = ',
                'key = 0x',
                'value = 0x',
                '[1] = ',
                '[2] = ',
                '[3] = '])
        self.expect(
            'frame variable dictionary --ptr-depth 1 --dynamic-type no-run-target',
            substrs=[
                '3 key/value pairs',
                '@"bar"',
                '@"2 elements"',
                '@"baz"',
                '2 key/value pairs'])
        self.expect(
            'frame variable mutabledict --ptr-depth 1 --dynamic-type no-run-target',
            substrs=[
                '4 key/value pairs',
                '(int)23',
                '@"123"',
                '@"http://www.apple.com"',
                '@"sourceofstuff"',
                '3 key/value pairs'])
        self.expect(
            'frame variable mutabledict --ptr-depth 2 --dynamic-type no-run-target',
            substrs=[
                '4 key/value pairs',
                '(int)23',
                '@"123"',
                '@"http://www.apple.com"',
                '@"sourceofstuff"',
                '3 key/value pairs',
                '@"bar"',
                '@"2 elements"'])
        self.expect(
            'frame variable mutabledict --ptr-depth 3 --dynamic-type no-run-target',
            substrs=[
                '4 key/value pairs',
                '(int)23',
                '@"123"',
                '@"http://www.apple.com"',
                '@"sourceofstuff"',
                '3 key/value pairs',
                '@"bar"',
                '@"2 elements"',
                '(int)1',
                '@"two"'])

        self.assertTrue(
            self.frame().FindVariable("dictionary").MightHaveChildren(),
            "dictionary says it does not have children!")
        self.assertTrue(
            self.frame().FindVariable("mutabledict").MightHaveChildren(),
            "mutable says it does not have children!")
