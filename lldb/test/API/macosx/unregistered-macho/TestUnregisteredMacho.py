"""Test that debugserver will parse a mach-o in inferior memory even if it's not loaded."""

import json
import os
import re
import subprocess

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class TestUnregisteredMacho(TestBase):
    # newer debugserver required for jGetLoadedDynamicLibrariesInfos
    # to support this
    @no_debug_info_test
    @requireDarwin
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )

        frame = thread.GetFrameAtIndex(0)
        macho_buf = frame.GetValueForVariablePath("macho_buf")
        macho_addr = macho_buf.GetValueAsUnsigned()
        invalid_macho_addr = macho_buf.GetValueAsUnsigned() + 4
        gdb_packet = (
            "process plugin packet send 'jGetLoadedDynamicLibrariesInfos:{\"solib_addresses\":[%d]}]'"
            % macho_addr
        )

        # Send the jGetLoadedDynamicLibrariesInfos packet
        # to debugserver, asking it to parse the mach-o binary
        # at this address and give us the UUID etc, even though
        # dyld doesn't think there is a binary at that address.
        # We won't get a pathname for the binary (from dyld), but
        # we will get to the LC_UUID and include that.
        self.expect(
            gdb_packet,
            substrs=['"pathname":""', '"uuid":"1B4E28BA-2FA1-11D2-883F-B9A761BDE3FB"'],
        )

        no_macho_gdb_packet = (
            "process plugin packet send 'jGetLoadedDynamicLibrariesInfos:{\"solib_addresses\":[%d]}]'"
            % invalid_macho_addr
        )
        self.expect(no_macho_gdb_packet, substrs=['response: {"images":[]}'])

        # Test that we get back the information for the properly
        # formatted Mach-O binary in memory, but do not get an
        # entry for the invalid Mach-O address.
        both_gdb_packet = (
            "process plugin packet send 'jGetLoadedDynamicLibrariesInfos:{\"solib_addresses\":[%d,%d]}]'"
            % (macho_addr, invalid_macho_addr)
        )
        self.expect(both_gdb_packet, substrs=['"load_address":%d,' % macho_addr])
        self.expect(
            both_gdb_packet,
            substrs=['"load_address":%d,' % invalid_macho_addr],
            matching=False,
        )

    def start_load_command_fixture(self):
        self.build()
        target, process, thread, _ = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c")
        )
        return target, process

    def image_address(self, target, name):
        value = target.FindFirstGlobalVariable(name)
        self.assertTrue(value.IsValid(), name)
        return value.GetValueAsUnsigned()

    def image_info(self, address):
        request = "jGetLoadedDynamicLibrariesInfos:" + json.dumps(
            {"solib_addresses": [address]}, separators=(",", ":")
        )
        return json.loads(lldbutil.send_packet_get_reply(self, request))["images"]

    def check_prefixes(self, image, version):
        self.assertEqual(image["pathname"], "")
        self.assertEqual(image["uuid"], "12345678-1234-5678-ABCD-0123456789AB")
        self.assertEqual(image["mach_header"]["magic"], 0xFEEDFACF)
        # CPU_SUBTYPE_X86_64_ALL is 3; CPU_SUBTYPE_ARM64_ALL is 0.
        subtype = 3 if "x86_64" in self.getArchitecture() else 0
        self.assertEqual(image["mach_header"]["cpusubtype"], subtype)
        self.assertEqual(image["min_version_os_name"], "macosx")
        self.assertEqual(image["min_version_os_sdk"], version)
        self.assertEqual(
            image["segments"],
            [
                {
                    "name": "1234567890abcdef",
                    "vmaddr": 0x123456789ABCDEF0,
                    "vmsize": 0x10000,
                    "fileoff": 0x1000,
                    "filesize": 0x2000,
                    "maxprot": 7,
                },
                {
                    "name": "__TEXT32",
                    "vmaddr": 0x5000,
                    "vmsize": 0x1000,
                    "fileoff": 0x20,
                    "filesize": 0x200,
                    "maxprot": 5,
                },
            ],
        )

    @no_debug_info_test
    @skipUnlessDarwin
    def test_load_command_prefixes_across_cache_boundaries(self):
        target, process = self.start_load_command_fixture()
        for name, version in [
            ("crossing_image", "13.2.1"),
            ("large_image", "10.12"),
            ("unreadable_unknown_image", "13.2.1"),
        ]:
            address = self.image_address(target, name)
            images = self.image_info(address)
            self.assertEqual(len(images), 1, name)
            self.assertEqual(images[0]["load_address"], address)
            self.check_prefixes(images[0], version)

        # MachProcess::ReadMemory must restore software breakpoint bytes in
        # cached command data, just as it does for an individual memory read.
        address = self.image_address(target, "breakpoint_address")
        bpsize = 4 if "arm64" in self.getArchitecture() else 1
        self.assertEqual(
            lldbutil.send_packet_get_reply(self, f"Z0,{address:x},{bpsize}"), "OK"
        )
        try:
            images = self.image_info(self.image_address(target, "crossing_image"))
            self.check_prefixes(images[0], "13.2.1")
        finally:
            self.assertEqual(
                lldbutil.send_packet_get_reply(self, f"z0,{address:x},{bpsize}"), "OK"
            )

    @no_debug_info_test
    @skipUnlessDarwin
    def test_unreadable_optional_load_command_payloads(self):
        target, process = self.start_load_command_fixture()
        for name in [
            "unreadable_uuid_image",
            "unreadable_version_image",
            "huge_size_image",
        ]:
            images = self.image_info(self.image_address(target, name))
            self.assertEqual(len(images), 1, name)
            self.assertEqual(images[0]["segments"], [])
            self.assertNotIn("min_version_os_name", images[0])
        images = self.image_info(self.image_address(target, "unreadable_segment_image"))
        self.assertEqual(images, [])

    @no_debug_info_test
    @skipUnlessDarwin
    def test_malformed_load_command_prefixes(self):
        target, process = self.start_load_command_fixture()
        for name in [
            "invalid_size_image",
            "invalid_count_image",
            "invalid_prefix_image",
            "invalid_overrun_image",
        ]:
            self.assertEqual(
                self.image_info(self.image_address(target, name)), [], name
            )
        self.assertEqual(self.image_info((1 << 64) - 8), [])
