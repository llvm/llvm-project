"""
Test lldb-vscode terminated event
"""

import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import re
import json

class TestVSCode_eventStatistic(lldbvscode_testcase.VSCodeTestCaseBase):

    def check_statistic(self, statistics):
        self.assertTrue(statistics['totalDebugInfoByteSize'] > 0)
        self.assertTrue(statistics['totalDebugInfoEnabled'] > 0)
        self.assertTrue(statistics['totalModuleCountHasDebugInfo'] > 0)

        self.assertIsNotNone(statistics['memory'])
        self.assertNotIn('modules', statistics.keys())

    def check_target(self, statistics):
        # lldb-vscode debugs one target at a time
        target = json.loads(statistics['targets'])[0]
        self.assertTrue(target['totalBreakpointResolveTime'] > 0)

        breakpoints = target['breakpoints']
        self.assertIn('foo',
                      breakpoints[0]['details']['Breakpoint']['BKPTResolver']['Options']['SymbolNames'],
                      'foo is a symbol breakpoint')
        self.assertTrue(breakpoints[1]['details']['Breakpoint']['BKPTResolver']['Options']['FileName'].endswith('main.cpp'),
                        'target has source line breakpoint in main.cpp')

    @skipIfWindows
    @skipIfRemote
    def test_terminated_event(self):
        '''
            Terminated Event
            Now contains the statistics of a debug session:
            metatdata:
                totalDebugInfoByteSize > 0
                totalDebugInfoEnabled > 0
                totalModuleCountHasDebugInfo > 0
                ...
            targetInfo:
                totalBreakpointResolveTime > 0
            breakpoints:
                recognize function breakpoint
                recognize source line breakpoint
            It should contains the breakpoints info: function bp & source line bp
        '''

        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)
        # Set breakpoints
        functions = ['foo']
        breakpoint_ids = self.set_function_breakpoints(functions)
        self.assertEquals(len(breakpoint_ids), len(functions), 'expect one breakpoint')
        main_bp_line = line_number('main.cpp', '// main breakpoint 1')
        breakpoint_ids.append(self.set_source_breakpoints('main.cpp', [main_bp_line]))

        self.continue_to_breakpoints(breakpoint_ids)
        self.continue_to_exit()

        statistics = self.vscode.wait_for_terminated()['statistics']
        self.check_statistic(statistics)
        self.check_target(statistics)

    @skipIfWindows
    @skipIfRemote
    def test_initialized_event(self):
        '''
            Initialized Event
            Now contains the statistics of a debug session:
                totalDebugInfoByteSize > 0
                totalDebugInfoEnabled > 0
                totalModuleCountHasDebugInfo > 0
                totalBreakpointResolveTime > 0
                ...
        '''

        program_basename = "a.out.stripped"
        program = self.getBuildArtifact(program_basename)
        self.build_and_launch(program)
        # Set breakpoints
        functions = ['foo']
        breakpoint_ids = self.set_function_breakpoints(functions)
        self.assertEquals(len(breakpoint_ids), len(functions), 'expect one breakpoint')
        main_bp_line = line_number('main.cpp', '// main breakpoint 1')
        breakpoint_ids.append(self.set_source_breakpoints('main.cpp', [main_bp_line]))

        self.continue_to_breakpoints(breakpoint_ids)
        statistics = self.vscode.initialized_event['statistics']
        self.check_statistic(statistics)
        self.continue_to_exit()
