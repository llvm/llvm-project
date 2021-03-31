"""
Test the 'memory read' command.
"""



import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil


class MemoryReadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def test_memory_read(self):
        """Test the 'memory read' command with plain and vector formats."""
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

        # Test the memory read commands.

        # (lldb) memory read -f d -c 1 `&argc`
        # 0x7fff5fbff9a0: 1
        self.runCmd("memory read -f d -c 1 `&argc`")

        # Find the starting address for variable 'argc' to verify later that the
        # '--format uint32_t[] --size 4 --count 4' option increments the address
        # correctly.
        line = self.res.GetOutput().splitlines()[0]
        items = line.split(':')
        address = int(items[0], 0)
        argc = int(items[1], 0)
        self.assertGreater(address, 0)
        self.assertEquals(argc, 1)

        # (lldb) memory read --format uint32_t[] --size 4 --count 4 `&argc`
        # 0x7fff5fbff9a0: {0x00000001}
        # 0x7fff5fbff9a4: {0x00000000}
        # 0x7fff5fbff9a8: {0x0ec0bf27}
        # 0x7fff5fbff9ac: {0x215db505}
        self.runCmd(
            "memory read --format uint32_t[] --size 4 --count 4 `&argc`")
        lines = self.res.GetOutput().splitlines()
        for i in range(4):
            if i == 0:
                # Verify that the printout for argc is correct.
                self.assertEqual(
                    argc, int(lines[i].split(':')[1].strip(' {}'), 0))
            addr = int(lines[i].split(':')[0], 0)
            # Verify that the printout for addr is incremented correctly.
            self.assertEqual(addr, (address + i * 4))

        # (lldb) memory read --format char[] --size 7 --count 1 `&my_string`
        # 0x7fff5fbff990: {abcdefg}
        self.expect(
            "memory read --format char[] --size 7 --count 1 `&my_string`",
            substrs=['abcdefg'])

        # (lldb) memory read --format 'hex float' --size 16 `&argc`
        # 0x7fff5fbff5b0: error: unsupported byte size (16) for hex float
        # format
        self.expect(
            "memory read --format 'hex float' --size 16 `&argc`",
            substrs=['unsupported byte size (16) for hex float format'])

        self.expect(
            "memory read --format 'float' --count 1 --size 8 `&my_double`",
            substrs=['1234.'])

        # (lldb) memory read --format 'float' --count 1 --size 20 `&my_double`
        # 0x7fff5fbff598: error: unsupported byte size (20) for float format
        self.expect(
            "memory read --format 'float' --count 1 --size 20 `&my_double`",
            substrs=['unsupported byte size (20) for float format'])

        self.expect('memory read --type int --count 5 `&my_ints[0]`',
                    substrs=['(int) 0x', '2', '4', '6', '8', '10'])

        self.expect(
            'memory read --type int --count 5 --format hex `&my_ints[0]`',
            substrs=[
                '(int) 0x',
                '0x',
                '0a'])

        self.expect(
            'memory read --type int --count 5 --offset 5 `&my_ints[0]`',
            substrs=[
                '(int) 0x',
                '12',
                '14',
                '16',
                '18',
                '20'])

        # the gdb format specifier and the size in characters for
        # the returned values including the 0x prefix.
        variations = [['b', 4], ['h', 6], ['w', 10], ['g', 18]]
        for v in variations:
          formatter = v[0]
          expected_object_length = v[1]
          self.runCmd(
              "memory read --gdb-format 4%s &my_uint64s" % formatter)
          lines = self.res.GetOutput().splitlines()
          objects_read = []
          for l in lines:
              objects_read.extend(l.split(':')[1].split())
          # Check that we got back 4 0x0000 etc bytes
          for o in objects_read:
              self.assertEqual(len(o), expected_object_length)
          self.assertEquals(len(objects_read), 4)
