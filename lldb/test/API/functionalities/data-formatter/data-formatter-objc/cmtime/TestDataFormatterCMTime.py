# encoding: utf-8

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class CMTimeDataFormatterTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    @skipUnlessDarwin
    def test_nsindexpath_with_run_command(self):
        """Test formatters for CMTime."""
        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"),
                    CURRENT_EXECUTABLE_SET)

        line = line_number('main.m', '// break here')
        lldbutil.run_break_set_by_file_and_line(
            self, "main.m", line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped', 'stop reason = breakpoint'])

        self.expect(
            'frame variable t1',
            substrs=[
                '1 half seconds', 'value = 1', 'timescale = 2', 'epoch = 0'
            ])
        self.expect(
            'frame variable t2',
            substrs=[
                '1 third of a second', 'value = 1', 'timescale = 3',
                'epoch = 0'
            ])
        self.expect(
            'frame variable t3',
            substrs=[
                '1 10th of a second', 'value = 1', 'timescale = 10',
                'epoch = 0'
            ])
        self.expect(
            'frame variable t4',
            substrs=['10 seconds', 'value = 10', 'timescale = 1', 'epoch = 0'])
        self.expect('frame variable t5', substrs=['+oo'])
        self.expect('frame variable t6', substrs=['-oo'])
        self.expect('frame variable t7', substrs=['indefinite'])
