"""
Test completion in our IOHandlers.
"""

import os

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbpexpect import PExpectTest


class BreakpointCallbackCommandSource(PExpectTest):

    mydir = TestBase.compute_mydir(__file__)
    file_to_source = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'source.lldb')

    # PExpect uses many timeouts internally and doesn't play well
    # under ASAN on a loaded machine..
    @skipIfAsan
    @skipIfEditlineSupportMissing
    @skipIf(oslist=["linux"], archs=["arm", "aarch64"])
    @skipIf(oslist=["freebsd"], bugnumber="llvm.org/pr48316")
    def test_breakpoint_callback_command_source(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        self.launch(exe)
        self.expect("b main", substrs=["Breakpoint 1"])
        self.child.send("breakpoint command add -s python\n")
        self.child.send(
            "frame.GetThread().GetProcess().GetTarget().GetDebugger().HandleCommand('command source -s true {}')\n"
            .format(self.file_to_source))
        self.child.send("DONE\n")
        self.expect_prompt()
        self.expect("run", substrs=["Process", "stopped"])
        self.expect("script print(foo)", substrs=["95126"])
