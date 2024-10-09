import lldb
from lldbsuite.test.decorators import *
import lldbsuite.test.lldbtest as lldbtest
import lldbsuite.test.lldbutil as lldbutil


class TestCase(lldbtest.TestBase):

    mydir = lldbtest.TestBase.compute_mydir(__file__)

    def read_ptr_from_memory(self, process, addr):
        error = lldb.SBError()
        value = process.ReadPointerFromMemory(addr, error)
        self.assertSuccess(error, "Failed to read memory")
        return value

    # Check that the CFA chain is correctly built
    def check_cfas(self, async_frames, process):
        async_cfas = list(map(lambda frame: frame.GetCFA(), async_frames))
        expected_cfas = [async_cfas[0]]
        # The CFA chain ends in nullptr.
        while expected_cfas[-1] != 0:
            expected_cfas.append(self.read_ptr_from_memory(process, expected_cfas[-1]))

        self.assertEqual(async_cfas, expected_cfas[:-1])

    def check_pcs(self, async_frames, process, target):
        for idx, frame in enumerate(async_frames[:-1]):
            # Read the continuation pointer from the second field of the CFA.
            continuation_ptr = self.read_ptr_from_memory(
                process, frame.GetCFA() + target.addr_size
            )

            # The PC of the previous frame should be the continuation pointer
            # with the funclet's prologue skipped.
            parent_frame = async_frames[idx + 1]
            prologue_to_skip = parent_frame.GetFunction().GetPrologueByteSize()
            self.assertEqual(continuation_ptr + prologue_to_skip, parent_frame.GetPC())


    def check_variables(self, async_frames, expected_values):
        for frame, expected_value in zip(async_frames, expected_values):
            myvar = frame.FindVariable("myvar")
            lldbutil.check_variable(self, myvar, False, value=expected_value)

    @swiftTest
    @skipIf(oslist=["windows", "linux"])
    def test(self):
        """Test `frame variable` in async functions"""
        self.build()

        source_file = lldb.SBFileSpec("main.swift")
        target, process, _, _ = lldbutil.run_to_source_breakpoint(
            self, "breakpoint1", source_file
        )

        async_frames = process.GetSelectedThread().frames
        self.check_cfas(async_frames, process)
        self.check_pcs(async_frames, process, target)
        self.check_variables(async_frames, ["222", "333", "444", "555"])

        target.DeleteAllBreakpoints()
        target.BreakpointCreateBySourceRegex("breakpoint2", source_file)
        process.Continue()
        # First frame is from a synchronous function
        frames = process.GetSelectedThread().frames
        async_frames = frames[1:]
        self.check_cfas(async_frames, process)
        self.check_pcs(async_frames, process, target)
        self.check_variables(async_frames, ["111", "222", "333", "444", "555"])

        # Now stop at the Q funclet right after the await to ASYNC___1
        target.DeleteAllBreakpoints()
        target.BreakpointCreateByName("$s1a12ASYNC___2___SiyYaFTQ0_")
        process.Continue()
        async_frames = process.GetSelectedThread().frames
        self.check_cfas(async_frames, process)
        self.check_pcs(async_frames, process, target)
        self.check_variables(async_frames, ["222", "333", "444", "555"])

        target.DeleteAllBreakpoints()
        target.BreakpointCreateBySourceRegex("breakpoint3", source_file)
        process.Continue()
        async_frames = process.GetSelectedThread().frames
        self.check_cfas(async_frames, process)
        self.check_pcs(async_frames, process, target)
        self.check_variables(async_frames, ["222", "333", "444", "555"])
