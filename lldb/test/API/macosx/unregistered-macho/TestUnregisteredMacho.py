"""Test that debugserver will parse a mach-o in inferior memory even if it's not loaded."""

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
    @skipIfOutOfTreeDebugserver  
    @no_debug_info_test
    @skipUnlessDarwin
    def test(self):
        self.build()
        target, process, thread, bkpt = lldbutil.run_to_source_breakpoint(
            self, "// break here", lldb.SBFileSpec("main.c"))

        frame = thread.GetFrameAtIndex(0)
        macho_buf = frame.GetValueForVariablePath("macho_buf")
        macho_addr = macho_buf.GetValueAsUnsigned()
        invalid_macho_addr = macho_buf.GetValueAsUnsigned() + 4
        gdb_packet = "process plugin packet send 'jGetLoadedDynamicLibrariesInfos:{\"solib_addresses\":[%d]}]'" % macho_addr

        # Send the jGetLoadedDynamicLibrariesInfos packet
        # to debugserver, asking it to parse the mach-o binary
        # at this address and give us the UUID etc, even though
        # dyld doesn't think there is a binary at that address.
        # We won't get a pathname for the binary (from dyld), but
        # we will get to the LC_UUID and include that.
        self.expect (gdb_packet, substrs=['"pathname":""', '"uuid":"1B4E28BA-2FA1-11D2-883F-B9A761BDE3FB"'])

        no_macho_gdb_packet = "process plugin packet send 'jGetLoadedDynamicLibrariesInfos:{\"solib_addresses\":[%d]}]'" % invalid_macho_addr
        self.expect (no_macho_gdb_packet, substrs=['response: {"images":[]}'])

        # Test that we get back the information for the properly
        # formatted Mach-O binary in memory, but do not get an
        # entry for the invalid Mach-O address.
        both_gdb_packet = "process plugin packet send 'jGetLoadedDynamicLibrariesInfos:{\"solib_addresses\":[%d,%d]}]'" % (macho_addr, invalid_macho_addr)
        self.expect (both_gdb_packet, substrs=['"load_address":%d,' % macho_addr])
        self.expect (both_gdb_packet, substrs=['"load_address":%d,' % invalid_macho_addr], matching=False)

