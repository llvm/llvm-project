"""
Test the 'register' command.
"""

from __future__ import print_function


import os
import sys
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):

    mydir = TestBase.compute_mydir(__file__)
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.has_teardown = False

    def tearDown(self):
        self.dbg.GetSelectedTarget().GetProcess().Destroy()
        TestBase.tearDown(self)

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'arm', 'i386', 'x86_64']))
    @expectedFailureAll(oslist=["freebsd", "netbsd"],
                        bugnumber='llvm.org/pr48371')
    def test_register_commands(self):
        """Test commands related to registers, in particular vector registers."""
        self.build()
        self.common_setup()

        # verify that logging does not assert
        self.log_enable("registers")

        self.expect("register read -a", MISSING_EXPECTED_REGISTERS,
                    substrs=['registers were unavailable'], matching=False)

        if self.getArchitecture() in ['amd64', 'i386', 'x86_64']:
            self.runCmd("register read xmm0")
            self.runCmd("register read ymm15")  # may be available
            self.runCmd("register read bnd0")  # may be available
        elif self.getArchitecture() in ['arm', 'armv7', 'armv7k', 'arm64', 'arm64e', 'arm64_32']:
            self.runCmd("register read s0")
            self.runCmd("register read q15")  # may be available

        self.expect(
            "register read -s 4",
            substrs=['invalid register set index: 4'],
            error=True)

    @skipIfiOSSimulator
    # Writing of mxcsr register fails, presumably due to a kernel/hardware
    # problem
    @skipIfTargetAndroid(archs=["i386"])
    @skipIf(archs=no_match(['amd64', 'arm', 'i386', 'x86_64']))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37995")
    def test_fp_register_write(self):
        """Test commands that write to registers, in particular floating-point registers."""
        self.build()
        self.fp_register_write()

    @skipIfiOSSimulator
    # "register read fstat" always return 0xffff
    @expectedFailureAndroid(archs=["i386"])
    @skipIf(archs=no_match(['amd64', 'i386', 'x86_64']))
    @skipIfOutOfTreeDebugserver
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37995")
    def test_fp_special_purpose_register_read(self):
        """Test commands that read fpu special purpose registers."""
        self.build()
        self.fp_special_purpose_register_read()

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'arm', 'i386', 'x86_64']))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37683")
    def test_register_expressions(self):
        """Test expression evaluation with commands related to registers."""
        self.build()
        self.common_setup()

        if self.getArchitecture() in ['amd64', 'i386', 'x86_64']:
            gpr = "eax"
            vector = "xmm0"
        elif self.getArchitecture() in ['arm64', 'aarch64', 'arm64e', 'arm64_32']:
            gpr = "w0"
            vector = "v0"
        elif self.getArchitecture() in ['arm', 'armv7', 'armv7k']:
            gpr = "r0"
            vector = "q0"

        self.expect("expr/x $%s" % gpr, substrs=['unsigned int', ' = 0x'])
        self.expect("expr $%s" % vector, substrs=['vector_type'])
        self.expect(
            "expr (unsigned int)$%s[0]" %
            vector, substrs=['unsigned int'])

        if self.getArchitecture() in ['amd64', 'x86_64']:
            self.expect(
                "expr -- ($rax & 0xffffffff) == $eax",
                substrs=['true'])

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'x86_64']))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37683")
    def test_convenience_registers(self):
        """Test convenience registers."""
        self.build()
        self.convenience_registers()

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'x86_64']))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37683")
    def test_convenience_registers_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        self.build()
        self.convenience_registers_with_process_attach(test_16bit_regs=False)

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'x86_64']))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37683")
    def test_convenience_registers_16bit_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        self.build()
        self.convenience_registers_with_process_attach(test_16bit_regs=True)

    def common_setup(self):
        exe = self.getBuildArtifact("a.out")

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        lldbutil.run_break_set_by_symbol(
            self, "main", num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect("thread list", STOPPED_DUE_TO_BREAKPOINT,
                    substrs=['stopped', 'stop reason = breakpoint'])

    # platform specific logging of the specified category
    def log_enable(self, category):
        # This intentionally checks the host platform rather than the target
        # platform as logging is host side.
        self.platform = ""
        if (sys.platform.startswith("freebsd") or
                sys.platform.startswith("linux") or
                sys.platform.startswith("netbsd")):
            self.platform = "posix"

        if self.platform != "":
            self.log_file = self.getBuildArtifact('TestRegisters.log')
            self.runCmd(
                "log enable " +
                self.platform +
                " " +
                str(category) +
                " registers -v -f " +
                self.log_file,
                RUN_SUCCEEDED)
            if not self.has_teardown:
                def remove_log(self):
                    if os.path.exists(self.log_file):
                        os.remove(self.log_file)
                self.has_teardown = True
                self.addTearDownHook(remove_log)

    def write_and_read(self, frame, register, new_value, must_exist=True):
        value = frame.FindValue(register, lldb.eValueTypeRegister)
        if must_exist:
            self.assertTrue(
                value.IsValid(),
                "finding a value for register " +
                register)
        elif not value.IsValid():
            return  # If register doesn't exist, skip this test

        # Also test the 're' alias.
        self.runCmd("re write " + register + " \'" + new_value + "\'")
        self.expect(
            "register read " +
            register,
            substrs=[
                register +
                ' = ',
                new_value])

    # This test relies on ftag containing the 'abridged' value.  Linux
    # and *BSD targets have been ported to report the full value instead
    # consistently with GDB.  They are covered by the new-style
    # lldb/test/Shell/Register/x86*-fp-read.test.
    @skipUnlessDarwin
    def fp_special_purpose_register_read(self):
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process and stop.
        self.expect("run", PROCESS_STOPPED, substrs=['stopped'])

        # Check stop reason; Should be either signal SIGTRAP or EXC_BREAKPOINT
        output = self.res.GetOutput()
        matched = False
        substrs = [
            'stop reason = EXC_BREAKPOINT',
            'stop reason = signal SIGTRAP']
        for str1 in substrs:
            matched = output.find(str1) != -1
            with recording(self, False) as sbuf:
                print("%s sub string: %s" % ('Expecting', str1), file=sbuf)
                print("Matched" if matched else "Not Matched", file=sbuf)
            if matched:
                break
        self.assertTrue(matched, STOPPED_DUE_TO_SIGNAL)

        process = target.GetProcess()
        self.assertEqual(process.GetState(), lldb.eStateStopped,
                        PROCESS_STOPPED)

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        currentFrame = thread.GetFrameAtIndex(0)
        self.assertTrue(currentFrame.IsValid(), "current frame is valid")

        # Extract the value of fstat and ftag flag at the point just before
        # we start pushing floating point values on st% register stack
        value = currentFrame.FindValue("fstat", lldb.eValueTypeRegister)
        error = lldb.SBError()
        reg_value_fstat_initial = value.GetValueAsUnsigned(error, 0)

        self.assertSuccess(error, "reading a value for fstat")
        value = currentFrame.FindValue("ftag", lldb.eValueTypeRegister)
        error = lldb.SBError()
        reg_value_ftag_initial = value.GetValueAsUnsigned(error, 0)

        self.assertSuccess(error, "reading a value for ftag")
        fstat_top_pointer_initial = (reg_value_fstat_initial & 0x3800) >> 11

        # Execute 'si' aka 'thread step-inst' instruction 5 times and with
        # every execution verify the value of fstat and ftag registers
        for x in range(0, 5):
            # step into the next instruction to push a value on 'st' register
            # stack
            self.runCmd("si", RUN_SUCCEEDED)

            # Verify fstat and save it to be used for verification in next
            # execution of 'si' command
            if not (reg_value_fstat_initial & 0x3800):
                self.expect("register read fstat", substrs=[
                            'fstat' + ' = ', str("0x%0.4x" % ((reg_value_fstat_initial & ~(0x3800)) | 0x3800))])
                reg_value_fstat_initial = (
                    (reg_value_fstat_initial & ~(0x3800)) | 0x3800)
                fstat_top_pointer_initial = 7
            else:
                self.expect("register read fstat", substrs=[
                            'fstat' + ' = ', str("0x%0.4x" % (reg_value_fstat_initial - 0x0800))])
                reg_value_fstat_initial = (reg_value_fstat_initial - 0x0800)
                fstat_top_pointer_initial -= 1

            # Verify ftag and save it to be used for verification in next
            # execution of 'si' command
            self.expect(
                "register read ftag", substrs=[
                    'ftag' + ' = ', str(
                        "0x%0.4x" %
                        (reg_value_ftag_initial | (
                            1 << fstat_top_pointer_initial)))])
            reg_value_ftag_initial = reg_value_ftag_initial | (
                1 << fstat_top_pointer_initial)

    def fp_register_write(self):
        exe = self.getBuildArtifact("a.out")

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, stop at the entry point.
        error = lldb.SBError()
        flags = target.GetLaunchInfo().GetLaunchFlags()
        process = target.Launch(
                lldb.SBListener(),
                None, None, # argv, envp
                None, None, None, # stdin/out/err
                self.get_process_working_directory(),
                flags, # launch flags
                True,  # stop at entry
                error)
        self.assertSuccess(error, "Launch succeeds")

        self.assertEqual(
            process.GetState(), lldb.eStateStopped,
            PROCESS_STOPPED)

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        currentFrame = thread.GetFrameAtIndex(0)
        self.assertTrue(currentFrame.IsValid(), "current frame is valid")

        if self.getArchitecture() in ['amd64', 'i386', 'x86_64']:
            reg_list = [
                # reg          value        must-have
                ("fcw", "0x0000ff0e", False),
                ("fsw", "0x0000ff0e", False),
                ("ftw", "0x0000ff0e", False),
                ("ip", "0x0000ff0e", False),
                ("dp", "0x0000ff0e", False),
                ("mxcsr", "0x0000ff0e", False),
                ("mxcsrmask", "0x0000ff0e", False),
            ]

            st0regname = None
            # Darwin is using stmmN by default but support stN as an alias.
            # Therefore, we need to check for stmmN first.
            if currentFrame.FindRegister("stmm0").IsValid():
                st0regname = "stmm0"
            elif currentFrame.FindRegister("st0").IsValid():
                st0regname = "st0"
            if st0regname is not None:
                # reg          value
                # must-have
                reg_list.append(
                    (st0regname, "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00}", True))
                reg_list.append(
                    ("xmm0",
                     "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}",
                     True))
                reg_list.append(
                    ("xmm15",
                     "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                     False))
        elif self.getArchitecture() in ['arm64', 'aarch64', 'arm64e', 'arm64_32']:
            reg_list = [
                # reg      value
                # must-have
                ("fpsr", "0xfbf79f9f", True),
                ("s0", "1.25", True),
                ("s31", "0.75", True),
                ("d1", "123", True),
                ("d17", "987", False),
                ("v1", "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}", True),
                ("v14",
                 "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                 False),
            ]
        elif self.getArchitecture() in ['armv7'] and self.platformIsDarwin():
            reg_list = [
                # reg      value
                # must-have
                ("fpsr", "0xfbf79f9f", True),
                ("s0", "1.25", True),
                ("s31", "0.75", True),
                ("d1", "123", True),
                ("d17", "987", False),
                ("q1", "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}", True),
                ("q14",
                 "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                 False),
            ]
        elif self.getArchitecture() in ['arm', 'armv7k']:
            reg_list = [
                # reg      value
                # must-have
                ("fpscr", "0xfbf79f9f", True),
                ("s0", "1.25", True),
                ("s31", "0.75", True),
                ("d1", "123", True),
                ("d17", "987", False),
                ("q1", "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}", True),
                ("q14",
                 "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                 False),
            ]

        for (reg, val, must) in reg_list:
            self.write_and_read(currentFrame, reg, val, must)

        if self.getArchitecture() in ['amd64', 'i386', 'x86_64']:
            if st0regname is None:
                self.fail("st0regname could not be determined")
            self.runCmd(
                "register write " +
                st0regname +
                " \"{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}\"")
            self.expect(
                "register read " +
                st0regname +
                " --format f",
                substrs=[
                    st0regname +
                    ' = 0'])

            has_avx = False
            has_mpx = False
            # Returns an SBValueList.
            registerSets = currentFrame.GetRegisters()
            for registerSet in registerSets:
                if 'advanced vector extensions' in registerSet.GetName().lower():
                    has_avx = True
                # FreeBSD/NetBSD reports missing register sets differently
                # at the moment and triggers false positive here.
                # TODO: remove FreeBSD/NetBSD exception when we make unsupported
                # register groups correctly disappear.
                if ('memory protection extension' in registerSet.GetName().lower()
                        and self.getPlatform() not in ["freebsd", "netbsd"]):
                    has_mpx = True

            if has_avx:
                new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x0d 0x0e 0x0f}"
                self.write_and_read(currentFrame, "ymm0", new_value)
                self.write_and_read(currentFrame, "ymm7", new_value)
                self.expect("expr $ymm0", substrs=['vector_type'])
            else:
                self.runCmd("register read ymm0")

            if has_mpx:
                # Test write and read for bnd0.
                new_value_w = "{0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a 0x0b 0x0c 0x0d 0x0e 0x0f 0x10}"
                self.runCmd("register write bnd0 \'" + new_value_w + "\'")
                new_value_r = "{0x0807060504030201 0x100f0e0d0c0b0a09}"
                self.expect("register read bnd0", substrs = ['bnd0 = ', new_value_r])
                self.expect("expr $bnd0", substrs = ['vector_type'])

                # Test write and for bndstatus.
                new_value = "{0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08}"
                self.write_and_read(currentFrame, "bndstatus", new_value)
                self.expect("expr $bndstatus", substrs = ['vector_type'])
            else:
                self.runCmd("register read bnd0")

    def convenience_registers(self):
        """Test convenience registers."""
        self.common_setup()

        # The command "register read -a" does output a derived register like
        # eax...
        self.expect("register read -a", matching=True,
                    substrs=['eax'])

        # ...however, the vanilla "register read" command should not output derived registers like eax.
        self.expect("register read", matching=False,
                    substrs=['eax'])

        # Test reading of rax and eax.
        self.expect("register read rax eax",
                    substrs=['rax = 0x', 'eax = 0x'])

        # Now write rax with a unique bit pattern and test that eax indeed
        # represents the lower half of rax.
        self.runCmd("register write rax 0x1234567887654321")
        self.expect("register read rax 0x1234567887654321",
                    substrs=['0x1234567887654321'])

    def convenience_registers_with_process_attach(self, test_16bit_regs):
        """Test convenience registers after a 'process attach'."""
        exe = self.getBuildArtifact("a.out")

        # Spawn a new process
        pid = self.spawnSubprocess(exe, ['wait_for_attach']).pid

        if self.TraceOn():
            print("pid of spawned process: %d" % pid)

        self.runCmd("process attach -p %d" % pid)

        # Check that "register read eax" works.
        self.runCmd("register read eax")

        if self.getArchitecture() in ['amd64', 'x86_64']:
            self.expect("expr -- ($rax & 0xffffffff) == $eax",
                        substrs=['true'])

        if test_16bit_regs:
            self.expect("expr -- $ax == (($ah << 8) | $al)",
                        substrs=['true'])

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'arm', 'i386', 'x86_64']))
    def test_invalid_invocation(self):
        self.build()
        self.common_setup()

        self.expect("register read -a arg", error=True,
                    substrs=["the --all option can't be used when registers names are supplied as arguments"])

        self.expect("register read --set 0 r", error=True,
                    substrs=["the --set <set> option can't be used when registers names are supplied as arguments"])

        self.expect("register write a", error=True,
                    substrs=["register write takes exactly 2 arguments: <reg-name> <value>"])
        self.expect("register write a b c", error=True,
                    substrs=["register write takes exactly 2 arguments: <reg-name> <value>"])

    @skipIfiOSSimulator
    @skipIf(archs=no_match(['amd64', 'arm', 'i386', 'x86_64']))
    def test_write_unknown_register(self):
        self.build()
        self.common_setup()

        self.expect("register write blub 1", error=True,
                    substrs=["error: Register not found for 'blub'."])
