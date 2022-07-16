from __future__ import print_function

# lldb test suite imports
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import TestBase

# gdb-remote-specific imports
import lldbgdbserverutils
from gdbremote_testcase import GdbRemoteTestCaseBase

import binascii
import os
import stat
import struct
import typing


class GDBStat(typing.NamedTuple):
    st_dev: int
    st_ino: int
    st_mode: int
    st_nlink: int
    st_uid: int
    st_gid: int
    st_rdev: int
    st_size: int
    st_blksize: int
    st_blocks: int
    st_atime: int
    st_mtime: int
    st_ctime: int


def uint32_or_zero(x):
    return x if x < 2**32 else 0


def uint32_or_max(x):
    return x if x < 2**32 else 2**32 - 1


def uint32_trunc(x):
    return x & (2**32 - 1)


class TestGdbRemotePlatformFile(GdbRemoteTestCaseBase):

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_rdonly(self):
        self.vFile_test(read=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly(self):
        self.vFile_test(write=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_rdwr(self):
        self.vFile_test(read=True, write=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_append(self):
        self.vFile_test(write=True, append=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_rdwr_append(self):
        self.vFile_test(read=True, write=True, append=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_trunc(self):
        self.vFile_test(write=True, trunc=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_rdwr_trunc(self):
        self.vFile_test(read=True, write=True, trunc=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_creat(self):
        self.vFile_test(write=True, creat=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_creat_excl(self):
        self.vFile_test(write=True, creat=True, excl=True)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_fail(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        temp_path = self.getBuildArtifact("test")
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

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_wronly_creat_excl_fail(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        temp_file = self.getBuildArtifact("test")
        with open(temp_file, "wb"):
            pass

        # attempt to open the file with O_CREAT|O_EXCL
        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:open:%s,a01,0#00" % (
                binascii.b2a_hex(temp_file.encode()).decode(),),
             {"direction": "send",
             "regex": r"^\$F-1,[0-9a-fA-F]+#[0-9a-fA-F]{2}$"}],
            True)
        self.expect_gdbremote_sequence()

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_size(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        temp_path = self.getBuildArtifact("test")
        test_data = b"test data of some length"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(test_data)

        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:size:%s#00" % (
                binascii.b2a_hex(temp_path.encode()).decode(),),
             {"direction": "send",
             "regex": r"^\$F([0-9a-fA-F]+)+#[0-9a-fA-F]{2}$",
             "capture": {1: "size"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertEqual(int(context["size"], 16), len(test_data))

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_mode(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        temp_path = self.getBuildArtifact("test")
        test_mode = 0o751

        with open(temp_path, "wb") as temp_file:
            os.chmod(temp_file.fileno(), test_mode)

        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:mode:%s#00" % (
                binascii.b2a_hex(temp_path.encode()).decode(),),
             {"direction": "send",
             "regex": r"^\$F([0-9a-fA-F]+)+#[0-9a-fA-F]{2}$",
             "capture": {1: "mode"}}],
            True)
        context = self.expect_gdbremote_sequence()
        self.assertEqual(int(context["mode"], 16), test_mode)

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_mode_fail(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        temp_path = self.getBuildArtifact("nonexist")

        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:mode:%s#00" % (
                binascii.b2a_hex(temp_path.encode()).decode(),),
             {"direction": "send",
             "regex": r"^\$F-1,0*2+#[0-9a-fA-F]{2}$"}],
            True)
        self.expect_gdbremote_sequence()

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_exists(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        temp_path = self.getBuildArtifact("test")
        with open(temp_path, "wb"):
            pass

        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:exists:%s#00" % (
                binascii.b2a_hex(temp_path.encode()).decode(),),
             "send packet: $F,1#00"],
            True)
        self.expect_gdbremote_sequence()

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_exists_not(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        test_path = self.getBuildArtifact("nonexist")
        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:exists:%s#00" % (
                binascii.b2a_hex(test_path.encode()).decode(),),
             "send packet: $F,0#00"],
            True)
        self.expect_gdbremote_sequence()

    @skipIfWindows
    @add_test_categories(["llgs"])
    def test_platform_file_fstat(self):
        server = self.connect_to_debug_monitor()
        self.assertIsNotNone(server)

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"some test data for stat")
            temp_file.flush()

            self.do_handshake()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:open:%s,0,0#00" % (
                    binascii.b2a_hex(temp_file.name.encode()).decode(),),
                 {"direction": "send",
                 "regex": r"^\$F([0-9a-fA-F]+)#[0-9a-fA-F]{2}$",
                 "capture": {1: "fd"}}],
                True)

            context = self.expect_gdbremote_sequence()
            self.assertIsNotNone(context)
            fd = int(context["fd"], 16)

            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:fstat:%x#00" % (fd,),
                 {"direction": "send",
                 "regex": r"^\$F([0-9a-fA-F]+);(.*)#[0-9a-fA-F]{2}$",
                 "capture": {1: "size", 2: "data"}}],
                True)
            context = self.expect_gdbremote_sequence()
            self.assertEqual(int(context["size"], 16), 64)
            # NB: we're using .encode() as a hack because the test suite
            # is wrongly using (unicode) str instead of bytes
            gdb_stat = GDBStat(
                *struct.unpack(">IIIIIIIQQQIII",
                               self.decode_gdbremote_binary(context["data"])
                                .encode("iso-8859-1")))
            sys_stat = os.fstat(temp_file.fileno())

            self.assertEqual(gdb_stat.st_dev, uint32_or_zero(sys_stat.st_dev))
            self.assertEqual(gdb_stat.st_ino, uint32_or_zero(sys_stat.st_ino))
            self.assertEqual(gdb_stat.st_mode, uint32_trunc(sys_stat.st_mode))
            self.assertEqual(gdb_stat.st_nlink, uint32_or_max(sys_stat.st_nlink))
            self.assertEqual(gdb_stat.st_uid, uint32_or_zero(sys_stat.st_uid))
            self.assertEqual(gdb_stat.st_gid, uint32_or_zero(sys_stat.st_gid))
            self.assertEqual(gdb_stat.st_rdev, uint32_or_zero(sys_stat.st_rdev))
            self.assertEqual(gdb_stat.st_size, sys_stat.st_size)
            self.assertEqual(gdb_stat.st_blksize, sys_stat.st_blksize)
            self.assertEqual(gdb_stat.st_blocks, sys_stat.st_blocks)
            self.assertEqual(gdb_stat.st_atime,
                             uint32_or_zero(int(sys_stat.st_atime)))
            self.assertEqual(gdb_stat.st_mtime,
                             uint32_or_zero(int(sys_stat.st_mtime)))
            self.assertEqual(gdb_stat.st_ctime,
                             uint32_or_zero(int(sys_stat.st_ctime)))

            self.reset_test_sequence()
            self.test_sequence.add_log_lines(
                ["read packet: $vFile:close:%x#00" % (fd,),
                 "send packet: $F0#00"],
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

        old_umask = os.umask(0o22)
        try:
            server = self.connect_to_debug_monitor()
        finally:
            os.umask(old_umask)
        self.assertIsNotNone(server)

        # create a temporary file with some data
        temp_path = self.getBuildArtifact("test")
        test_data = 'some test data longer than 16 bytes\n'

        if creat:
            self.assertFalse(os.path.exists(temp_path))
        else:
            with open(temp_path, "wb") as temp_file:
                temp_file.write(test_data.encode())

        # open the file for reading
        self.do_handshake()
        self.test_sequence.add_log_lines(
            ["read packet: $vFile:open:%s,%x,1a0#00" % (
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
            with open(temp_path, "rb") as temp_file:
                if creat:
                    self.assertEqual(os.fstat(temp_file.fileno()).st_mode & 0o7777,
                                     0o640)
                data = test_data.encode()
                if trunc or creat:
                    data = b"\0" * 6 + b"somedata"
                elif append:
                    data += b"somedata"
                else:
                    data = data[:6] + b"somedata" + data[6 + 8:]
                self.assertEqual(temp_file.read(), data)
