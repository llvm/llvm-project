from __future__ import print_function

# lldb test suite imports
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase

# gdb-remote-specific imports
import lldbgdbserverutils
from gdbremote_testcase import GdbRemoteTestCaseBase

import binascii
import stat
import tempfile


class TestGdbRemotePlatformFile(GdbRemoteTestCaseBase):

    mydir = TestBase.compute_mydir(__file__)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_rdonly(self):
        self.vFile_test(read=True)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_wronly(self):
        self.vFile_test(write=True)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_rdwr(self):
        self.vFile_test(read=True, write=True)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_append(self):
        self.vFile_test(write=True, append=True)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_rdwr_append(self):
        self.vFile_test(read=True, write=True, append=True)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_trunc(self):
        self.vFile_test(write=True, trunc=True)

    @expectedFailureAll(oslist=["windows"])
    @add_test_categories(["llgs"])
    def test_platform_file_rdwr_trunc(self):
        self.vFile_test(read=True, write=True, trunc=True)

    @add_test_categories(["llgs"])
    def test_platform_file_wronly_creat(self):
        self.vFile_test(write=True, creat=True)

    @add_test_categories(["llgs"])
    def test_platform_file_wronly_creat_excl(self):
        self.vFile_test(write=True, creat=True, excl=True)

    @add_test_categories(["llgs"])
    def test_platform_file_wronly_fail(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        # create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "test")
            self.assertFalse(os.path.exists(temp_path))

            # attempt to open the file without O_CREAT
            self.do_handshake()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:open:%s,1,0#00" % (
                    binascii.b2a_hex(temp_path.encode()).decode(),),
                 {"direction": "send",
                 "regex": r"^\$F-1,[0-9a-fA-F]+#[0-9a-fA-F]{2}$"}],
                True)
            self.expect_gdbremote_sequence()

    @add_test_categories(["llgs"])
    def test_platform_file_wronly_creat_excl_fail(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        with tempfile.NamedTemporaryFile() as temp_file:
            # attempt to open the file with O_CREAT|O_EXCL
            self.do_handshake()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:open:%s,a01,0#00" % (
                    binascii.b2a_hex(temp_file.name.encode()).decode(),),
                 {"direction": "send",
                 "regex": r"^\$F-1,[0-9a-fA-F]+#[0-9a-fA-F]{2}$"}],
                True)
            self.expect_gdbremote_sequence()

    def expect_error(self):
        self.test_sequence.add_log_lines(
            [{"direction": "send",
             "regex": r"^\$F-1,[0-9a-fA-F]+#[0-9a-fA-F]{2}$"}],
            True)
        self.expect_gdbremote_sequence()

    def vFile_test(self, read=False, write=False, append=False, trunc=False,
                   creat=False, excl=False):
        if read and write:
            mode = 2
        elif write:
            mode = 1
        else:  # read
            mode = 0
        if append:
            mode |= 8
        if creat:
            mode |= 0x200
        if trunc:
            mode |= 0x400
        if excl:
            mode |= 0x800

        old_umask = os.umask(0)
        try:
            server = self.connect_to_debug_monitor()
        finally:
            os.umask(old_umask)
        self.assertIsNotNone(server)

        # create a temporary file with some data
        test_data = 'some test data longer than 16 bytes\n'
        if creat:
            temp_dir = tempfile.TemporaryDirectory()
        else:
            temp_file = tempfile.NamedTemporaryFile()

        try:
            if creat:
                temp_path = os.path.join(temp_dir.name, "test")
                self.assertFalse(os.path.exists(temp_path))
            else:
                temp_file.write(test_data.encode())
                temp_file.flush()
                temp_path = temp_file.name

            # open the file for reading
            self.do_handshake()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:open:%s,%x,1b6#00" % (
                    binascii.b2a_hex(temp_path.encode()).decode(),
                    mode),
                 {"direction": "send",
                 "regex": r"^\$F([0-9a-fA-F]+)#[0-9a-fA-F]{2}$",
                 "capture": {1: "fd"}}],
                True)

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)
            fd = int(context["fd"], 16)

            # read data from the file
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:pread:%x,11,10#00" % (fd,)],
                True)
            if read:
                self.test_sequence.add_log_lines(
                    [{"direction": "send",
                     "regex": r"^\$F([0-9a-fA-F]+);(.*)#[0-9a-fA-F]{2}$",
                     "capture": {1: "size", 2: "data"}}],
                    True)
                context = self.expect_gdbremote_sequence()
                self.assertIsNotNone(context)
                if trunc:
                    self.assertEqual(context["size"], "0")
                    self.assertEqual(context["data"], "")
                else:
                    self.assertEqual(context["size"], "11")  # hex
                    self.assertEqual(context["data"], test_data[0x10:0x10 + 0x11])
            else:
                self.expect_error()

            # another offset
            if read and not trunc:
                self.reset_test_sequence()
                self.test_sequence.add_log_lines(
                    ["read packet: $vFile:pread:%x,6,3#00" % (fd,),
                     {"direction": "send",
                     "regex": r"^\$F([0-9a-fA-F]+);(.+)#[0-9a-fA-F]{2}$",
                     "capture": {1: "size", 2: "data"}}],
                    True)
                context = self.expect_gdbremote_sequence()
                self.assertIsNotNone(context)
                self.assertEqual(context["size"], "6")  # hex
                self.assertEqual(context["data"], test_data[3:3 + 6])

            # write data to the file
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:pwrite:%x,6,somedata#00" % (fd,)],
                True)
            if write:
                self.test_sequence.add_log_lines(
                    ["send packet: $F8#00"],
                    True)
                self.expect_gdbremote_sequence()
            else:
                self.expect_error()

            # close the file
            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:close:%x#00" % (fd,),
                 "send packet: $F0#00"],
                True)
            self.expect_gdbremote_sequence()

            if write:
                # check if the data was actually written
                if creat:
                    temp_file = open(temp_path, "rb")
                    self.assertEqual(os.fstat(temp_file.fileno()).st_mode & 0o7777,
                                     0o666)
                temp_file.seek(0)
                data = test_data.encode()
                if trunc or creat:
                    data = b"\0" * 6 + b"somedata"
                elif append:
                    data += b"somedata"
                else:
                    data = data[:6] + b"somedata" + data[6 + 8:]
                self.assertEqual(temp_file.read(), data)
        finally:
            if creat:
                temp_dir.cleanup()
            else:
                temp_file.close()
