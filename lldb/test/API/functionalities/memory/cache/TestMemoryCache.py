"""
Test the MemoryCache L1 flush.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class MemoryCacheTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    @skipIfWindows # This is flakey on Windows: llvm.org/pr38373
    def test_memory_cache(self):
        """Test the MemoryCache class with a sequence of 'memory read' and 'memory write' operations."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main() after the variables are assigned values.
        lldbutil.run_break_set_by_file_and_line(
            self, "main.cpp", self.line, num_expected_locations=1, loc_exact=True)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

        # The breakpoint should have a hit count of 1.
        self.expect("breakpoint list -f", BREAKPOINT_HIT_ONCE,
                    substrs=[' resolved, hit count = 1'])

        # Read a chunk of memory containing &my_ints[0]. The number of bytes read
        # must be greater than m_L2_cache_line_byte_size to make sure the L1
        # cache is used.
        self.runCmd('memory read -f d -c 201 `&my_ints - 100`')

        # Check the value of my_ints[0] is the same as set in main.cpp.
        line = self.res.GetOutput().splitlines()[100]
        self.assertEquals(0x00000042, int(line.split(':')[1], 0))

        # Change the value of my_ints[0] in memory.
        self.runCmd("memory write -s 4 `&my_ints` AA")

        # Re-read the chunk of memory. The cache line should have been
        # flushed because of the 'memory write'.
        self.runCmd('memory read -f d -c 201 `&my_ints - 100`')

        # Check the value of my_ints[0] have been updated correctly.
        line = self.res.GetOutput().splitlines()[100]
        self.assertEquals(0x000000AA, int(line.split(':')[1], 0))
