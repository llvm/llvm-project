"""
Test the lldb disassemble command on lib stdc++.
"""

import os
import re
import lldb
from lldbsuite.test.lldbtest import *
import lldbsuite.test.lldbutil as lldbutil
from lldbsuite.test.decorators import *


class StdCXXDisassembleTestCase(TestBase):
    @skipIfWindows
    def test_stdcxx_disasm(self):
        """Do 'disassemble' on each and every 'Code' symbol entry from the std c++ lib."""
        self.build()
        (target, process, thread, bkpt) = lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line", lldb.SBFileSpec("main.cpp")
        )

        # Disassemble the functions on the call stack.
        self.runCmd("thread backtrace")
        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertIsNotNone(thread)
        depth = thread.GetNumFrames()
        for i in range(depth - 1):
            frame = thread.GetFrameAtIndex(i)
            function = frame.GetFunction()
            if function.GetName():
                self.runCmd("disassemble -n '%s'" % function.GetName())

        lib_stdcxx = "FAILHORRIBLYHERE"
        # Find the stdc++ library...
        stdlib_regex = re.compile(r"/lib(std)?c\+\+")
        for module in target.module[stdlib_regex]:
            lib_stdcxx = module.file.fullpath
            break

        # At this point, lib_stdcxx is the full path to the stdc++ library and
        # module is the corresponding SBModule.

        if "lib" not in lib_stdcxx:
            self.skipTest(
                "This test requires libstdc++.so or libc++.dylib in the target's module list."
            )

        self.runCmd("image dump symtab '%s'" % lib_stdcxx)
        raw_output = self.res.GetOutput()
        # Now, look for every 'Code' symbol and feed its load address into the
        # command: 'disassemble -s load_address -e end_address', where the
        # end_address is taken from the next consecutive 'Code' symbol entry's
        # load address.
        #
        # The load address column comes after the file address column, with both
        # looks like '0xhhhhhhhh', i.e., 8 hexadecimal digits.
        codeRE = re.compile(
            r"""
                             \ Code\ {9}      # ' Code' followed by 9 SPCs,
                             0x[0-9a-f]{16}   # the file address column, and
                             \                # a SPC, and
                             (0x[0-9a-f]{16}) # the load address column, and
                             .*               # the rest.
                             """,
            re.VERBOSE,
        )
        # Maintain a start address variable; if we arrive at a consecutive Code
        # entry, then the load address of the that entry is fed as the end
        # address to the 'disassemble -s SA -e LA' command.
        SA = None
        for line in raw_output.split(os.linesep):
            match = codeRE.search(line)
            if match:
                LA = match.group(1)
                if self.TraceOn():
                    print("line:", line)
                    print("load address:", LA)
                    print("SA:", SA)
                if SA and LA:
                    if int(LA, 16) > int(SA, 16):
                        self.runCmd("disassemble -s %s -e %s" % (SA, LA))
                SA = LA
            else:
                # This entry is not a Code entry.  Reset SA = None.
                SA = None
