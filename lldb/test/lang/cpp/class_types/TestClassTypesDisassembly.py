"""
Test the lldb disassemble command on each call frame when stopped on C's ctor.
"""

import os, time
import unittest2
import lldb
from lldbtest import *
import lldbutil

class IterateFrameAndDisassembleTestCase(TestBase):

    mydir = os.path.join("lang", "cpp", "class_types")

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @dsym_test
    def test_with_dsym_and_run_command(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.buildDsym()
        self.disassemble_call_stack()

    @dwarf_test
    def test_with_dwarf_and_run_command(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.buildDwarf()
        self.disassemble_call_stack()

    @unittest2.skipUnless(sys.platform.startswith("darwin"), "requires Darwin")
    @python_api_test
    @dsym_test
    def test_with_dsym_and_python_api(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.buildDsym()
        self.disassemble_call_stack_python()

    @python_api_test
    @dwarf_test
    def test_with_dwarf_and_python_api(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.buildDwarf()
        self.disassemble_call_stack_python()

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break for main.cpp.
        self.line = line_number('main.cpp', '// Set break point at this line.')

    def breakOnCtor(self):
        """Setup/run the program so it stops on C's constructor."""
        exe = os.path.join(os.getcwd(), "a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break on the ctor function of class C.
        lldbutil.run_break_set_by_file_and_line (self, "main.cpp", self.line, num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped',
                       'stop reason = breakpoint'])

        # We should be stopped on the ctor function of class C.
        self.expect("thread backtrace", BACKTRACE_DISPLAYED_CORRECTLY,
            substrs = ['C::C'])

    def disassemble_call_stack(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.breakOnCtor()

        raw_output = self.res.GetOutput()
        frameRE = re.compile(r"""
                              ^\s\sframe        # heading for the frame info,
                              .*                # wildcard, and
                              0x[0-9a-f]{16}    # the frame pc, and
                              \sa.out`(.+)      # module`function, and
                              \s\+\s            # the rest ' + ....'
                              """, re.VERBOSE)
        for line in raw_output.split(os.linesep):
            match = frameRE.search(line)
            if match:
                function = match.group(1)
                #print "line:", line
                #print "function:", function
                self.runCmd("disassemble -n '%s'" % function)

    def disassemble_call_stack_python(self):
        """Disassemble each call frame when stopped on C's constructor."""
        self.breakOnCtor()

        # Now use the Python API to get at each function on the call stack and
        # disassemble it.
        target = self.dbg.GetSelectedTarget()
        process = target.GetProcess()
        thread = process.GetThreadAtIndex(0)
        depth = thread.GetNumFrames()
        for i in range(depth - 1):
            frame = thread.GetFrameAtIndex(i)
            function = frame.GetFunction()
            # Print the function header.
            if self.TraceOn():
                print
                print function
            if function:
                # Get all instructions for this function and print them out.
                insts = function.GetInstructions(target)
                for inst in insts:
                    # We could simply do 'print inst' to print out the disassembly.
                    # But we want to print to stdout only if self.TraceOn() is True.
                    disasm = str(inst)
                    if self.TraceOn():
                        print disasm


if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
