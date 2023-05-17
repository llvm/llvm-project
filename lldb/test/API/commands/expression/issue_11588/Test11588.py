"""
Test the solution to issue 11581.
valobj.AddressOf() returns None when an address is
expected in a SyntheticChildrenProvider
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class Issue11581TestCase(TestBase):

    @skipIfWindows #This test is now flaky on windows, see llvm.org/pr24778
    def test_11581_commands(self):
        # This is the function to remove the custom commands in order to have a
        # clean slate for the next test case.
        def cleanup():
            self.runCmd('type synthetic clear', check=False)

        # Execute the cleanup function during test case tear down.
        self.addTearDownHook(cleanup)

        """valobj.AddressOf() should return correct values."""
        self.build()

        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(self,
                                              'Set breakpoint here.',
                                              lldb.SBFileSpec("main.cpp", False))
        self.runCmd("command script import --allow-reload s11588.py")
        self.runCmd(
            "type synthetic add --python-class s11588.Issue11581SyntheticProvider StgClosure")

        self.expect("expr --show-types -- *((StgClosure*)(r14-1))",
                    substrs=["(StgClosure) $",
                             "(StgClosure *) &$", "0x",
                             "addr = ",
                             "load_address = "])

        # register r14 is an x86_64 extension let's skip this part of the test
        # if we are on a different architecture
        if self.getArchitecture() == 'x86_64':
            target = self.dbg.GetSelectedTarget()
            process = target.GetProcess()
            frame = process.GetSelectedThread().GetSelectedFrame()
            pointer = frame.FindVariable("r14")
            addr = pointer.GetValueAsUnsigned(0)
            self.assertNotEqual(addr, 0, "could not read pointer to StgClosure")
            addr = addr - 1
            self.runCmd("register write r14 %d" % addr)
            self.expect(
                "register read r14", substrs=[
                    "0x", hex(addr)[
                        2:].rstrip("L")])  # Remove trailing 'L' if it exists
            self.expect("expr --show-types -- *(StgClosure*)$r14",
                        substrs=["(StgClosure) $",
                                 "(StgClosure *) &$", "0x",
                                 hex(addr)[2:].rstrip("L"),
                                 "addr = ",
                                 str(addr),
                                 "load_address = ",
                                 str(addr)])
