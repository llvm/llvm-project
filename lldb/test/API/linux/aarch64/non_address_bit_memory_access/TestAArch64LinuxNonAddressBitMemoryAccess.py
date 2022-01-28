"""
Test that lldb removes non-address bits in situations where they would cause
failures if not removed. Like when reading memory. Tests are done at command
and API level because commands may remove non-address bits for display
reasons which can make it seem like the operation as a whole works but at the
API level it won't if we don't remove them there also.
"""



import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class AArch64LinuxNonAddressBitMemoryAccessTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)

    NO_DEBUG_INFO_TESTCASE = True

    def setup_test(self):
        if not self.isAArch64PAuth():
            self.skipTest('Target must support pointer authentication.')

        self.build()
        self.runCmd("file " + self.getBuildArtifact("a.out"), CURRENT_EXECUTABLE_SET)

        lldbutil.run_break_set_by_file_and_line(self, "main.c",
            line_number('main.c', '// Set break point at this line.'),
            num_expected_locations=1)

        self.runCmd("run", RUN_SUCCEEDED)

        if self.process().GetState() == lldb.eStateExited:
            self.fail("Test program failed to run.")

        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs=['stopped',
                     'stop reason = breakpoint'])

    def check_cmd_read_write(self, write_to, read_from, data):
        self.runCmd("memory write {} {}".format(write_to, data))
        self.expect("memory read {}".format(read_from),
                substrs=[data])

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_non_address_bit_memory_read_write_cmds(self):
        self.setup_test()

        # Writes should be visible through either pointer
        self.check_cmd_read_write("buf", "buf", "01 02 03 04")
        self.check_cmd_read_write("buf_with_non_address", "buf_with_non_address", "02 03 04 05")
        self.check_cmd_read_write("buf", "buf_with_non_address", "03 04 05 06")
        self.check_cmd_read_write("buf_with_non_address", "buf", "04 05 06 07")

        # Printing either should get the same result
        self.expect("expression -f hex -- *(uint32_t*)buf", substrs=["0x07060504"])
        self.expect("expression -f hex -- *(uint32_t*)buf_with_non_address",
                    substrs=["0x07060504"])

    def get_ptr_values(self):
        frame  = self.process().GetThreadAtIndex(0).GetFrameAtIndex(0)
        buf = frame.FindVariable("buf").GetValueAsUnsigned()
        buf_with_non_address = frame.FindVariable("buf_with_non_address").GetValueAsUnsigned()
        return buf, buf_with_non_address

    def check_api_read_write(self, write_to, read_from, data):
        error = lldb.SBError()
        written = self.process().WriteMemory(write_to, data, error)
        self.assertTrue(error.Success())
        self.assertEqual(len(data), written)
        buf_content = self.process().ReadMemory(read_from, 4, error)
        self.assertTrue(error.Success())
        self.assertEqual(data, buf_content)

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_non_address_bit_memory_read_write_api_process(self):
        self.setup_test()
        buf, buf_with_non_address = self.get_ptr_values()

        # Writes are visible through either pointer
        self.check_api_read_write(buf, buf, bytes([0, 1, 2, 3]))
        self.check_api_read_write(buf_with_non_address, buf_with_non_address, bytes([1, 2, 3, 4]))
        self.check_api_read_write(buf, buf_with_non_address, bytes([2, 3, 4, 5]))
        self.check_api_read_write(buf_with_non_address, buf, bytes([3, 4, 5, 6]))

        # Now check all the "Read<type>FromMemory" don't fail
        error = lldb.SBError()
        # Last 4 bytes are just for the pointer read
        data = bytes([0x4C, 0x4C, 0x44, 0x42, 0x00, 0x12, 0x34, 0x56])
        written = self.process().WriteMemory(buf, data, error)
        self.assertTrue(error.Success())
        self.assertEqual(len(data), written)

        # C string
        c_string = self.process().ReadCStringFromMemory(buf_with_non_address, 5, error)
        self.assertTrue(error.Success())
        self.assertEqual("LLDB", c_string)

        # Unsigned
        unsigned_num = self.process().ReadUnsignedFromMemory(buf_with_non_address, 4, error)
        self.assertTrue(error.Success())
        self.assertEqual(0x42444c4c, unsigned_num)

        # Pointer
        ptr = self.process().ReadPointerFromMemory(buf_with_non_address, error)
        self.assertTrue(error.Success())
        self.assertEqual(0x5634120042444c4c, ptr)

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_non_address_bit_memory_read_write_api_target(self):
        self.setup_test()
        buf, buf_with_non_address = self.get_ptr_values()

        # Target only has ReadMemory
        error = lldb.SBError()
        data = bytes([1, 2, 3, 4])
        written = self.process().WriteMemory(buf, data, error)
        self.assertTrue(error.Success())
        self.assertEqual(len(data), written)

        addr = lldb.SBAddress()
        addr.SetLoadAddress(buf, self.target())
        buf_read = self.target().ReadMemory(addr, 4, error)
        self.assertTrue(error.Success())
        self.assertEqual(data, buf_read)

        addr.SetLoadAddress(buf_with_non_address, self.target())
        buf_non_address_read = self.target().ReadMemory(addr, 4, error)
        self.assertTrue(error.Success())
        self.assertEqual(data, buf_non_address_read)

        # Read<type>FromMemory are in Target but not SBTarget so no tests for those.

    @skipUnlessArch("aarch64")
    @skipUnlessPlatform(["linux"])
    def test_non_address_bit_memory_caching(self):
        # The read/write tests above do exercise the cache but this test
        # only cares that the cache sees buf and buf_with_non_address
        # as the same location.
        self.setup_test()
        buf, buf_with_non_address = self.get_ptr_values()

        # Enable packet logging so we can see when reads actually
        # happen.
        log_file = self.getBuildArtifact("lldb-non-address-bit-log.txt")
        # This defaults to overwriting the file so we don't need to delete
        # any existing files.
        self.runCmd("log enable gdb-remote packets -f '%s'" % log_file)

        # This should fill the cache by doing a read of buf_with_non_address
        # with the non-address bits removed (which is == buf).
        self.runCmd("p buf_with_non_address")
        # This will read from the cache since the two pointers point to the
        # same place.
        self.runCmd("p buf")

        # Open log ignoring utf-8 decode errors
        with open(log_file, 'r', errors='ignore') as f:
            read_packet = "send packet: $x{:x}"
            read_buf_packet = read_packet.format(buf)
            read_buf_with_non_address_packet = read_packet.format(buf_with_non_address)

            # We expect to find 1 and only 1 read of buf.
            # We expect to find no reads using buf_with_no_address.
            found_read_buf = False
            for line in f:
                if read_buf_packet in line:
                    if found_read_buf:
                        self.fail("Expected 1 read of buf but found more than one.")
                    found_read_buf = True

                if read_buf_with_non_address_packet in line:
                    self.fail("Unexpected read of buf_with_non_address found.")

            if not found_read_buf:
                self.fail("Did not find any reads of buf.")
