"""
Test the solution to issue 11581.
valobj.AddressOf() returns None when an address is
expected in a SyntheticChildrenProvider
"""

import os, time
import unittest2
import lldb
from lldbtest import *

class Issue11581TestCase(TestBase):

    mydir = os.path.join("expression_command", "issue_11588")

    def test_11581_commands(self):
        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type synthetic clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """valobj.AddressOf() should return correct values."""
        self.buildDefault()

        self.runCmd("file a.out", CURRENT_EXECUTABLE_SET)

        self.runCmd("breakpoint set --name main")

        self.runCmd("run", RUN_SUCCEEDED)

        self.runCmd("next", RUN_SUCCEEDED)
        self.runCmd("next", RUN_SUCCEEDED)
        self.runCmd("next", RUN_SUCCEEDED)
        self.runCmd("next", RUN_SUCCEEDED)
        self.runCmd("next", RUN_SUCCEEDED)

        self.runCmd("command script import --allow-reload s11588.py")
        self.runCmd("type synthetic add --python-class s11588.Issue11581SyntheticProvider StgClosure")

        self.expect("print *((StgClosure*)(r14-1))",
            substrs = ["(StgClosure) $",
            "(StgClosure *) &$","0x",
            "addr = ",
            "load_address = "])


        target = lldb.debugger.GetSelectedTarget()
        process = target.GetProcess()
# register r14 does not exist on 32-bit architectures, it is an x86_64 extension
# let's skip this part of the test if we are in 32-bit mode
        if process.GetAddressByteSize() == 8:
                frame = process.GetSelectedThread().GetSelectedFrame()
                pointer = frame.FindVariable("r14")
                addr = pointer.GetValueAsUnsigned(0)
                self.assertTrue(addr != 0, "could not read pointer to StgClosure")
                addr = addr - 1
                self.runCmd("register write r14 %d" % addr)
                self.expect("register read r14",
                    substrs = ["0x",hex(addr)[2:].rstrip("L")])  # Remove trailing 'L' if it exists
                self.expect("print *(StgClosure*)$r14",
                    substrs = ["(StgClosure) $",
                    "(StgClosure *) &$","0x",
                    "(long) addr = ",
                    "(long) load_address = ",
                    hex(addr)[2:].rstrip("L"),
                    str(addr)])


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
