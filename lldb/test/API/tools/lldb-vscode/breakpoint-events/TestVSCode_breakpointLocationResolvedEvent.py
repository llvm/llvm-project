"""
Test lldb-vscode setBreakpoints request
"""


import unittest2
import vscode
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil
import lldbvscode_testcase
import os


class TestVSCode_breakpointLocationResolvedEvent(lldbvscode_testcase.VSCodeTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    def build_launch_and_attach(self):
        self.build_and_create_debug_adaptor()
        # launch externally
        exe = self.getBuildArtifact("dylib_loader")
        popen = self.spawnSubprocess(exe)
        # attach
        self.attach(exe, popen.pid)

    def set_breakpoint(self, filename, comment):
        source_basename = filename
        source_path = os.path.join(os.getcwd(), source_basename)
        bp_line = line_number(filename, comment)
        return self.vscode.request_setBreakpoints(source_path,
                                                      [bp_line])

    @skipUnlessPlatform(["linux"])
    def test_breakpoint_location_resolved_event(self):
        '''
            This test sets a breakpoint in a shared library before it's loaded.
            This will make the client receive a breakpoint notification of
            unresolved location. Once the library is loaded the client should
            receive another change event indicating the location is resolved.
        '''
        self.build_launch_and_attach()
        self.set_breakpoint('dylib_loader.c', 'break after dlopen')
        response = self.set_breakpoint('dylib.c', 'breakpoint dylib')
        if response:
            breakpoints = response['body']['breakpoints']
            for breakpoint in breakpoints:
                bp_id = breakpoint['id']
                self.assertFalse(breakpoint['verified'],
                                "expect dylib breakpoint to be unverified")
                break
        response = self.vscode.request_evaluate("flip_to_1_to_continue = 1")
        self.assertTrue(response['success'])

        self.continue_to_next_stop()
        self.assertTrue(len(self.vscode.breakpoint_events) > 1,
                        "make sure we got a breakpoint event")

        # find the last breakpoint event for bp_id
        for event in reversed(self.vscode.breakpoint_events):
            if event['body']['breakpoint']['id'] == bp_id:
                break
        body = event['body']
        # Verify the details of the breakpoint changed notification.
        self.assertTrue(body['reason'] == 'changed',
                "breakpoint event reason should be changed")
        breakpoint = body['breakpoint']
        self.assertTrue(breakpoint['verified'] == True,
                "breakpoint event should be verified")
        self.assertTrue(breakpoint['id'] == bp_id,
                "breakpoint event is for breakpoint %i" % (bp_id))
