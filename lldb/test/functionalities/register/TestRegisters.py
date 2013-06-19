"""
Test the 'register' command.
"""

import os, sys, time
import re
import unittest2
import lldb
from lldbtest import *
import lldbutil

class RegisterCommandsTestCase(TestBase):

    mydir = os.path.join("functionalities", "register")

    def setUp(self):
        TestBase.setUp(self)
        self.has_teardown = False

    def test_register_commands(self):
        """Test commands related to registers, in particular vector registers."""
        if not self.getArchitecture() in ['i386', 'x86_64']:
            self.skipTest("This test requires i386 or x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.register_commands()

    def test_fp_register_write(self):
        """Test commands that write to registers, in particular floating-point registers."""
        if not self.getArchitecture() in ['i386', 'x86_64']:
            self.skipTest("This test requires i386 or x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.fp_register_write()

    def test_register_expressions(self):
        """Test expression evaluation with commands related to registers."""
        if not self.getArchitecture() in ['i386', 'x86_64']:
            self.skipTest("This test requires i386 or x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.register_expressions()

    def test_convenience_registers(self):
        """Test convenience registers."""
        if not self.getArchitecture() in ['x86_64']:
            self.skipTest("This test requires x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.convenience_registers()

    @skipIfLinux # Expression evaluation fails after attach by pid
    def test_convenience_registers_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        if not self.getArchitecture() in ['x86_64']:
            self.skipTest("This test requires x86_64 as the architecture for the inferior")
        self.buildDefault()
        self.convenience_registers_with_process_attach()

    def common_setup(self):
        exe = os.path.join(os.getcwd(), "a.out")

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        lldbutil.run_break_set_by_symbol (self, "main", num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
            substrs = ['stopped', 'stop reason = breakpoint'])

    # platform specific logging of the specified category
    def log_enable(self, category):
        self.platform = ""
        if sys.platform.startswith("darwin"):
            self.platform = "" # TODO: add support for "log enable darwin registers"
        if sys.platform.startswith("linux"):
            self.platform = "linux"

        if self.platform != "":
            log_file = os.path.join(os.getcwd(), 'TestRegisters.log')
            self.runCmd("log enable " + self.platform + " " + str(category) + " registers -v -f " + log_file, RUN_SUCCEEDED)
            if not self.has_teardown:
                self.has_teardown = True
                self.addTearDownHook(lambda: os.remove(log_file))

    def register_commands(self):
        """Test commands related to registers, in particular vector registers."""
        self.common_setup()

        # verify that logging does not assert
        self.log_enable("registers")

        self.expect("register read -a", MISSING_EXPECTED_REGISTERS,
            substrs = ['registers were unavailable'], matching = False)
        self.runCmd("register read xmm0")
        self.runCmd("register read ymm15") # may be available

        self.expect("register read -s 3",
            substrs = ['invalid register set index: 3'], error = True)

    def write_and_restore(self, frame, register, must_exist = True):
        value = frame.FindValue(register, lldb.eValueTypeRegister)
        if must_exist:
            self.assertTrue(value.IsValid(), "finding a value for register " + register)
        elif not value.IsValid():
            return # If register doesn't exist, skip this test

        error = lldb.SBError()
        register_value = value.GetValueAsUnsigned(error, 0)
        self.assertTrue(error.Success(), "reading a value for " + register)

        self.runCmd("register write " + register + " 0xff0e")
        self.expect("register read " + register,
            substrs = [register + ' = 0x', 'ff0e'])

        self.runCmd("register write " + register + " " + str(register_value))
        self.expect("register read " + register,
            substrs = [register + ' = 0x'])

    def vector_write_and_read(self, frame, register, new_value, must_exist = True):
        value = frame.FindValue(register, lldb.eValueTypeRegister)
        if must_exist:
            self.assertTrue(value.IsValid(), "finding a value for register " + register)
        elif not value.IsValid():
            return # If register doesn't exist, skip this test

        self.runCmd("register write " + register + " \'" + new_value + "\'")
        self.expect("register read " + register,
            substrs = [register + ' = ', new_value])

    def fp_register_write(self):
        exe = os.path.join(os.getcwd(), "a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        lldbutil.run_break_set_by_symbol (self, "main", num_expected_locations=-1)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        process = target.GetProcess()
        self.assertTrue(process.GetState() == lldb.eStateStopped,
                        PROCESS_STOPPED)

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        currentFrame = thread.GetFrameAtIndex(0)
        self.assertTrue(currentFrame.IsValid(), "current frame is valid")

        self.write_and_restore(currentFrame, "fcw", False)
        self.write_and_restore(currentFrame, "fsw", False)
        self.write_and_restore(currentFrame, "ftw", False)
        self.write_and_restore(currentFrame, "ip", False)
        self.write_and_restore(currentFrame, "dp", False)
        self.write_and_restore(currentFrame, "mxcsr", False)
        self.write_and_restore(currentFrame, "mxcsrmask", False)

        new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"
        self.vector_write_and_read(currentFrame, "stmm0", new_value)
        new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a}"
        self.vector_write_and_read(currentFrame, "stmm7", new_value)

        new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}"
        self.vector_write_and_read(currentFrame, "xmm0", new_value)
        new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}"
        self.vector_write_and_read(currentFrame, "xmm15", new_value, False)

        has_avx = False 
        registerSets = currentFrame.GetRegisters() # Returns an SBValueList.
        for registerSet in registerSets:
            if 'advanced vector extensions' in registerSet.GetName().lower():
                has_avx = True
                break

        if has_avx:
            new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x0d 0x0e 0x0f}"
            self.vector_write_and_read(currentFrame, "ymm0", new_value)
            self.vector_write_and_read(currentFrame, "ymm15", new_value)
            self.expect("expr $ymm0", substrs = ['vector_type'])
        else:
            self.runCmd("register read ymm0")

    def register_expressions(self):
        """Test expression evaluation with commands related to registers."""
        self.common_setup()

        self.expect("expr/x $eax",
            substrs = ['unsigned int', ' = 0x'])

        if self.getArchitecture() in ['x86_64']:
            self.expect("expr -- ($rax & 0xffffffff) == $eax",
                substrs = ['true'])

        self.expect("expr $xmm0",
            substrs = ['vector_type'])

        self.expect("expr (unsigned int)$xmm0[0]",
            substrs = ['unsigned int'])

    def convenience_registers(self):
        """Test convenience registers."""
        self.common_setup()

        # The command "register read -a" does output a derived register like eax...
        self.expect("register read -a", matching=True,
            substrs = ['eax'])

        # ...however, the vanilla "register read" command should not output derived registers like eax.
        self.expect("register read", matching=False,
            substrs = ['eax'])
        
        # Test reading of rax and eax.
        self.expect("register read rax eax",
            substrs = ['rax = 0x', 'eax = 0x'])

        # Now write rax with a unique bit pattern and test that eax indeed represents the lower half of rax.
        self.runCmd("register write rax 0x1234567887654321")
        self.expect("register read rax 0x1234567887654321",
            substrs = ['0x1234567887654321'])

    def convenience_registers_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        exe = self.lldbHere
        
        # Spawn a new process
        proc = self.spawnSubprocess(exe, [self.lldbOption])
        self.addTearDownHook(self.cleanupSubprocesses)

        if self.TraceOn():
            print "pid of spawned process: %d" % proc.pid

        self.runCmd("process attach -p %d" % proc.pid)

        # Check that "register read eax" works.
        self.runCmd("register read eax")

        if self.getArchitecture() in ['x86_64']:
            self.expect("expr -- ($rax & 0xffffffff) == $eax",
                substrs = ['true'])

        self.expect("expr -- $ax == (($ah << 8) | $al)",
            substrs = ['true'])

if __name__ == '__main__':
    import atexit
    lldb.SBDebugger.Initialize()
    atexit.register(lambda: lldb.SBDebugger.Terminate())
    unittest2.main()
