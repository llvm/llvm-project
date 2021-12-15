"""
Test the 'memory read' command.
"""

import lldb
import lldbsuite.test.lldbutil as lldbutil

from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *


class MemoryReadTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def build_run_stop(self):
        self.build()
        lldbutil.run_to_source_breakpoint(self, "// break here",
                lldb.SBFileSpec("main.c"))

    def test_memory_read_c_string(self):
        """Test that reading memory as a c string respects the size limit given
           and warns if the null terminator is missing."""
        self.build_run_stop()

        # The size here is the size in memory so it includes the null terminator.
        cmd = "memory read --format \"c-string\" --size {} &my_string"

        # Size matches the size of the array.
        self.expect(cmd.format(8), substrs=['\"abcdefg\"'])

        # If size would take us past the terminator we stop at the terminator.
        self.expect(cmd.format(10), substrs=['\"abcdefg\"'])

        # Size 3 means 2 chars and a terminator. So we print 2 chars but warn because
        # the third isn't 0 as expected.
        self.expect(cmd.format(3), substrs=['\"ab\"'])
        self.assertRegex(self.res.GetError(),
            "unable to find a NULL terminated string at 0x[0-9A-fa-f]+."
            " Consider increasing the maximum read length.")

    def test_memory_read(self):
        """Test the 'memory read' command with plain and vector formats."""
        self.build_run_stop()

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

    def test_memory_read_file(self):
        self.build_run_stop()
        res = lldb.SBCommandReturnObject()
        self.ci.HandleCommand("memory read -f d -c 1 `&argc`", res)
        self.assertTrue(res.Succeeded(), "memory read failed:" + res.GetError())

        # Record golden output.
        golden_output = res.GetOutput()

        memory_read_file = self.getBuildArtifact("memory-read-output")

        def check_file_content(expected):
            with open(memory_read_file) as f:
                lines = f.readlines()
                lines = [s.strip() for s in lines]
                expected = [s.strip() for s in expected]
                self.assertEqual(lines, expected)

        # Sanity check.
        self.runCmd("memory read -f d -c 1 -o '{}' `&argc`".format(memory_read_file))
        check_file_content([golden_output])

        # Write some garbage to the file.
        with open(memory_read_file, 'w') as f:
            f.write("some garbage")

        # Make sure the file is truncated when we run the command again.
        self.runCmd("memory read -f d -c 1 -o '{}' `&argc`".format(memory_read_file))
        check_file_content([golden_output])

        # Make sure the file is appended when we run the command with --append-outfile.
        self.runCmd(
            "memory read -f d -c 1 -o '{}' --append-outfile `&argc`".format(
                memory_read_file))
        check_file_content([golden_output, golden_output])
