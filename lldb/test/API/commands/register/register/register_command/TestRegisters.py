"""
Test the 'register' command.
"""

import os
import sys
import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class RegisterCommandsTestCase(TestBase):
    NO_DEBUG_INFO_TESTCASE = True

    def setUp(self):
        TestBase.setUp(self)
        self.has_teardown = False

    def tearDown(self):
        self.dbg.GetSelectedTarget().GetProcess().Destroy()
        TestBase.tearDown(self)

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "arm", "i386", "x86_64"]))
    @expectedFailureAll(oslist=["freebsd", "netbsd"], bugnumber="llvm.org/pr48371")
    def test_register_commands(self):
        """Test commands related to registers, in particular vector registers."""
        self.build()
        self.common_setup()

        # verify that logging does not assert
        self.log_enable("registers")

        self.expect(
            "register read -a",
            MISSING_EXPECTED_REGISTERS,
            substrs=["registers were unavailable"],
            matching=False,
        )

        all_registers = self.res.GetOutput()

        if self.getArchitecture() in ["amd64", "i386", "x86_64"]:
            self.runCmd("register read xmm0")
            if "ymm15 = " in all_registers:
                self.runCmd("register read ymm15")  # may be available
            if "bnd0 = " in all_registers:
                self.runCmd("register read bnd0")  # may be available
        elif self.getArchitecture() in [
            "arm",
            "armv7",
            "armv7k",
            "arm64",
            "arm64e",
            "arm64_32",
        ]:
            self.runCmd("register read s0")
            if "q15 = " in all_registers:
                self.runCmd("register read q15")  # may be available

        self.expect(
            "register read -s 4", substrs=["invalid register set index: 4"], error=True
        )

    @skipIfiOSSimulator
    # Writing of mxcsr register fails, presumably due to a kernel/hardware
    # problem
    @skipIfTargetAndroid(archs=["i386"])
    @skipIf(archs=no_match(["amd64", "arm", "i386", "x86_64"]))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37995")
    def test_fp_register_write(self):
        """Test commands that write to registers, in particular floating-point registers."""
        self.build()
        self.fp_register_write()

    @skipIfiOSSimulator
    # "register read fstat" always return 0xffff
    @expectedFailureAndroid(archs=["i386"])
    @skipIf(archs=no_match(["amd64", "i386", "x86_64"]))
    @skipIfOutOfTreeDebugserver
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37995")
    def test_fp_special_purpose_register_read(self):
        """Test commands that read fpu special purpose registers."""
        self.build()
        self.fp_special_purpose_register_read()

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "arm", "i386", "x86_64"]))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37683")
    def test_register_expressions(self):
        """Test expression evaluation with commands related to registers."""
        self.build()
        self.common_setup()

        if self.getArchitecture() in ["amd64", "i386", "x86_64"]:
            gpr = "eax"
            vector = "xmm0"
        elif self.getArchitecture() in ["arm64", "aarch64", "arm64e", "arm64_32"]:
            gpr = "w0"
            vector = "v0"
        elif self.getArchitecture() in ["arm", "armv7", "armv7k"]:
            gpr = "r0"
            vector = "q0"

        self.expect("expr/x $%s" % gpr, substrs=["unsigned int", " = 0x"])
        self.expect("expr $%s" % vector, substrs=["vector_type"])
        self.expect("expr (unsigned int)$%s[0]" % vector, substrs=["unsigned int"])

        if self.getArchitecture() in ["amd64", "x86_64"]:
            self.expect("expr -- ($rax & 0xffffffff) == $eax", substrs=["true"])

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "x86_64"]))
    @expectedFailureAll(oslist=["windows"], bugnumber="llvm.org/pr37683")
    def test_convenience_registers(self):
        """Test convenience registers."""
        self.build()
        self.convenience_registers()

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "x86_64"]))
    def test_convenience_registers_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        self.build()
        self.convenience_registers_with_process_attach(test_16bit_regs=False)

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "x86_64"]))
    def test_convenience_registers_16bit_with_process_attach(self):
        """Test convenience registers after a 'process attach'."""
        self.build()
        self.convenience_registers_with_process_attach(test_16bit_regs=True)

    def common_setup(self):
        exe = self.getBuildArtifact("a.out")

        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        # Break in main().
        lldbutil.run_break_set_by_symbol(self, "main", num_expected_locations=-1)

        self.runCmd("run", RUN_SUCCEEDED)

        # The stop reason of the thread should be breakpoint.
        self.expect(
            "thread list",
            STOPPED_DUE_TO_BREAKPOINT,
            substrs=["stopped", "stop reason = breakpoint"],
        )

    # platform specific logging of the specified category
    def log_enable(self, category):
        # This intentionally checks the host platform rather than the target
        # platform as logging is host side.
        self.platform = ""
        if (
            sys.platform.startswith("freebsd")
            or sys.platform.startswith("linux")
            or sys.platform.startswith("netbsd")
        ):
            self.platform = "posix"

        if self.platform != "":
            self.log_file = self.getBuildArtifact("TestRegisters.log")
            self.runCmd(
                "log enable "
                + self.platform
                + " "
                + str(category)
                + " registers -v -f "
                + self.log_file,
                RUN_SUCCEEDED,
            )
            if not self.has_teardown:

                def remove_log(self):
                    if os.path.exists(self.log_file):
                        os.remove(self.log_file)

                self.has_teardown = True
                self.addTearDownHook(remove_log)

    def write_and_read(self, frame, register, new_value, must_exist=True):
        value = frame.FindValue(register, lldb.eValueTypeRegister)
        if must_exist:
            self.assertTrue(value.IsValid(), "finding a value for register " + register)
        elif not value.IsValid():
            return  # If register doesn't exist, skip this test

        # Also test the 're' alias.
        self.runCmd("re write " + register + " '" + new_value + "'")
        self.expect("register read " + register, substrs=[register + " = ", new_value])

    # This test relies on ftag containing the 'abridged' value.  Linux
    # and *BSD targets have been ported to report the full value instead
    # consistently with GDB.  They are covered by the new-style
    # lldb/test/Shell/Register/x86*-fp-read.test.
    @skipUnlessDarwin
    def fp_special_purpose_register_read(self):
        target = self.createTestTarget()

        # Launch the process and stop.
        self.expect("run", PROCESS_STOPPED, substrs=["stopped"])

        # Check stop reason; Should be either signal SIGTRAP or EXC_BREAKPOINT
        output = self.res.GetOutput()
        matched = False
        substrs = ["stop reason = EXC_BREAKPOINT", "stop reason = signal SIGTRAP"]
        for str1 in substrs:
            matched = output.find(str1) != -1
            with recording(self, False) as sbuf:
                print("%s sub string: %s" % ("Expecting", str1), file=sbuf)
                print("Matched" if matched else "Not Matched", file=sbuf)
            if matched:
                break
        self.assertTrue(matched, STOPPED_DUE_TO_SIGNAL)

        process = target.GetProcess()
        self.assertState(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

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
                self.expect(
                    "register read fstat",
                    substrs=[
                        "fstat" + " = ",
                        str(
                            "0x%0.4x" % ((reg_value_fstat_initial & ~(0x3800)) | 0x3800)
                        ),
                    ],
                )
                reg_value_fstat_initial = (reg_value_fstat_initial & ~(0x3800)) | 0x3800
                fstat_top_pointer_initial = 7
            else:
                self.expect(
                    "register read fstat",
                    substrs=[
                        "fstat" + " = ",
                        str("0x%0.4x" % (reg_value_fstat_initial - 0x0800)),
                    ],
                )
                reg_value_fstat_initial = reg_value_fstat_initial - 0x0800
                fstat_top_pointer_initial -= 1

            # Verify ftag and save it to be used for verification in next
            # execution of 'si' command
            self.expect(
                "register read ftag",
                substrs=[
                    "ftag" + " = ",
                    str(
                        "0x%0.4x"
                        % (reg_value_ftag_initial | (1 << fstat_top_pointer_initial))
                    ),
                ],
            )
            reg_value_ftag_initial = reg_value_ftag_initial | (
                1 << fstat_top_pointer_initial
            )

    def fp_register_write(self):
        target = self.createTestTarget()

        # Launch the process, stop at the entry point.
        error = lldb.SBError()
        flags = target.GetLaunchInfo().GetLaunchFlags()
        process = target.Launch(
            lldb.SBListener(),
            None,
            None,  # argv, envp
            None,
            None,
            None,  # stdin/out/err
            self.get_process_working_directory(),
            flags,  # launch flags
            True,  # stop at entry
            error,
        )
        self.assertSuccess(error, "Launch succeeds")

        self.assertEqual(process.GetState(), lldb.eStateStopped, PROCESS_STOPPED)

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        currentFrame = thread.GetFrameAtIndex(0)
        self.assertTrue(currentFrame.IsValid(), "current frame is valid")

        if self.getArchitecture() in ["amd64", "i386", "x86_64"]:
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
                    (
                        st0regname,
                        "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x00 0x00}",
                        True,
                    )
                )
                reg_list.append(
                    (
                        "xmm0",
                        "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}",
                        True,
                    )
                )
                reg_list.append(
                    (
                        "xmm15",
                        "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                        False,
                    )
                )
        elif self.getArchitecture() in ["arm64", "aarch64", "arm64e", "arm64_32"]:
            reg_list = [
                # reg      value
                # must-have
                ("fpsr", "0xfbf79f9f", True),
                ("s0", "1.25", True),
                ("s31", "0.75", True),
                ("d1", "123", True),
                ("d17", "987", False),
                (
                    "v1",
                    "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}",
                    True,
                ),
                (
                    "v14",
                    "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                    False,
                ),
            ]
        elif self.getArchitecture() in ["armv7"] and self.platformIsDarwin():
            reg_list = [
                # reg      value
                # must-have
                ("fpsr", "0xfbf79f9f", True),
                ("s0", "1.25", True),
                ("s31", "0.75", True),
                ("d1", "123", True),
                ("d17", "987", False),
                (
                    "q1",
                    "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}",
                    True,
                ),
                (
                    "q14",
                    "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                    False,
                ),
            ]
        elif self.getArchitecture() in ["arm", "armv7k"]:
            reg_list = [
                # reg      value
                # must-have
                ("fpscr", "0xfbf79f9f", True),
                ("s0", "1.25", True),
                ("s31", "0.75", True),
                ("d1", "123", True),
                ("d17", "987", False),
                (
                    "q1",
                    "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x2f 0x2f}",
                    True,
                ),
                (
                    "q14",
                    "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f}",
                    False,
                ),
            ]

        for reg, val, must in reg_list:
            self.write_and_read(currentFrame, reg, val, must)

        if self.getArchitecture() in ["amd64", "i386", "x86_64"]:
            if st0regname is None:
                self.fail("st0regname could not be determined")
            self.runCmd(
                "register write "
                + st0regname
                + ' "{0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00}"'
            )
            self.expect(
                "register read " + st0regname + " --format f",
                substrs=[st0regname + " = 0"],
            )

            # Check if AVX/MPX registers are defined at all.
            registerSets = currentFrame.GetRegisters()
            registers = frozenset(
                reg.GetName() for registerSet in registerSets for reg in registerSet
            )
            has_avx_regs = "ymm0" in registers
            has_mpx_regs = "bnd0" in registers
            # Check if they are actually present.
            self.runCmd("register read -a")
            output = self.res.GetOutput()
            has_avx = "ymm0 =" in output
            has_mpx = "bnd0 =" in output

            if has_avx:
                new_value = "{0x01 0x02 0x03 0x00 0x00 0x00 0x00 0x00 0x09 0x0a 0x2f 0x2f 0x2f 0x2f 0x0e 0x0f 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x0d 0x0e 0x0f}"
                self.write_and_read(currentFrame, "ymm0", new_value)
                self.write_and_read(currentFrame, "ymm7", new_value)
                self.expect("expr $ymm0", substrs=["vector_type"])
            elif has_avx_regs:
                self.expect("register read ymm0", substrs=["error: unavailable"])
            else:
                self.expect(
                    "register read ymm0",
                    substrs=["Invalid register name 'ymm0'"],
                    error=True,
                )

            if has_mpx:
                # Test write and read for bnd0.
                new_value_w = "{0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08 0x09 0x0a 0x0b 0x0c 0x0d 0x0e 0x0f 0x10}"
                self.runCmd("register write bnd0 '" + new_value_w + "'")
                new_value_r = "{0x0807060504030201 0x100f0e0d0c0b0a09}"
                self.expect("register read bnd0", substrs=["bnd0 = ", new_value_r])
                self.expect("expr $bnd0", substrs=["vector_type"])

                # Test write and for bndstatus.
                new_value = "{0x01 0x02 0x03 0x04 0x05 0x06 0x07 0x08}"
                self.write_and_read(currentFrame, "bndstatus", new_value)
                self.expect("expr $bndstatus", substrs=["vector_type"])
            elif has_mpx_regs:
                self.expect("register read bnd0", substrs=["error: unavailable"])
            else:
                self.expect(
                    "register read bnd0",
                    substrs=["Invalid register name 'bnd0'"],
                    error=True,
                )

    def convenience_registers(self):
        """Test convenience registers."""
        self.common_setup()

        # The command "register read -a" does output a derived register like
        # eax...
        self.expect("register read -a", matching=True, substrs=["eax"])

        # ...however, the vanilla "register read" command should not output derived registers like eax.
        self.expect("register read", matching=False, substrs=["eax"])

        # Test reading of rax and eax.
        self.expect("register read rax eax", substrs=["rax = 0x", "eax = 0x"])

        # Now write rax with a unique bit pattern and test that eax indeed
        # represents the lower half of rax.
        self.runCmd("register write rax 0x1234567887654321")
        self.expect("register read rax", substrs=["0x1234567887654321"])

    def convenience_registers_with_process_attach(self, test_16bit_regs):
        """Test convenience registers after a 'process attach'."""
        exe = self.getBuildArtifact("a.out")

        # Spawn a new process
        pid = self.spawnSubprocess(exe, ["wait_for_attach"]).pid

        if self.TraceOn():
            print("pid of spawned process: %d" % pid)

        self.runCmd("process attach -p %d" % pid)

        # Check that "register read eax" works.
        self.runCmd("register read eax")

        if self.getArchitecture() in ["amd64", "x86_64"]:
            self.expect("expr -- ($rax & 0xffffffff) == $eax", substrs=["true"])

        if test_16bit_regs:
            self.expect("expr -- $ax == (($ah << 8) | $al)", substrs=["true"])

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "arm", "i386", "x86_64"]))
    def test_invalid_invocation(self):
        self.build()
        self.common_setup()

        self.expect(
            "register read -a arg",
            error=True,
            substrs=[
                "the --all option can't be used when registers names are supplied as arguments"
            ],
        )

        self.expect(
            "register read --set 0 r",
            error=True,
            substrs=[
                "the --set <set> option can't be used when registers names are supplied as arguments"
            ],
        )

        self.expect(
            "register write a",
            error=True,
            substrs=["register write takes exactly 2 arguments: <reg-name> <value>"],
        )
        self.expect(
            "register write a b c",
            error=True,
            substrs=["register write takes exactly 2 arguments: <reg-name> <value>"],
        )

    @skipIfiOSSimulator
    @skipIf(archs=no_match(["amd64", "arm", "i386", "x86_64"]))
    def test_write_unknown_register(self):
        self.build()
        self.common_setup()

        self.expect(
            "register write blub 1",
            error=True,
            substrs=["error: Register not found for 'blub'."],
        )

    def test_info_unknown_register(self):
        self.build()
        self.common_setup()

        self.expect(
            "register info blub",
            error=True,
            substrs=["error: No register found with name 'blub'."],
        )

    def test_info_many_registers(self):
        self.build()
        self.common_setup()

        # Only 1 register allowed at this time.
        self.expect(
            "register info abc def",
            error=True,
            substrs=["error: register info takes exactly 1 argument"],
        )

    @skipIf(archs=no_match(["aarch64"]))
    def test_info_register(self):
        # The behaviour of this command is generic but the specific registers
        # are not, so this is written for AArch64 only.
        # Text alignment and ordering are checked in the DumpRegisterInfo and
        # RegisterFlags unit tests.
        self.build()
        self.common_setup()

        # Standard register. Doesn't invalidate anything, doesn't have an alias.
        self.expect(
            "register info x1",
            substrs=[
                "Name: x1",
                "Size: 8 bytes (64 bits)",
                "In sets: General Purpose Registers",
            ],
        )
        self.expect(
            "register info x1", substrs=["Invalidates:", "Name: x1 ("], matching=False
        )

        # These registers invalidate others as they are subsets of those registers.
        self.expect("register info w1", substrs=["Invalidates: x1"])
        self.expect("register info s0", substrs=["Invalidates: v0, d0"])

        # This has an alternative name according to the ABI.
        self.expect("register info x30", substrs=["Name: lr (x30)"])

    @skipUnlessPlatform(["linux"])
    @skipIf(archs=no_match(["x86_64"]))
    def test_fs_gs_base(self):
        """
        Tests fs_base register can be read and equals to pthread_self() return value
        and gs_base register equals zero.
        """
        self.build()
        target = self.createTestTarget()
        # Launch the process and stop.
        self.expect("run", PROCESS_STOPPED, substrs=["stopped"])

        process = target.GetProcess()

        thread = process.GetThreadAtIndex(0)
        self.assertTrue(thread.IsValid(), "current thread is valid")

        current_frame = thread.GetFrameAtIndex(0)
        self.assertTrue(current_frame.IsValid(), "current frame is valid")

        reg_fs_base = current_frame.FindRegister("fs_base")
        self.assertTrue(reg_fs_base.IsValid(), "fs_base is not available")
        reg_gs_base = current_frame.FindRegister("gs_base")
        self.assertTrue(reg_gs_base.IsValid(), "gs_base is not available")
        self.assertEqual(reg_gs_base.GetValueAsSigned(-1), 0, f"gs_base should be zero")

        # Evaluate pthread_self() and compare against fs_base register read.
        pthread_self_code = "(uint64_t)pthread_self()"
        pthread_self_val = current_frame.EvaluateExpression(pthread_self_code)
        self.assertTrue(
            pthread_self_val.IsValid(), f"{pthread_self_code} evaluation has failed"
        )
        self.assertNotEqual(
            reg_fs_base.GetValueAsSigned(-1), -1, f"fs_base returned -1 which is wrong"
        )

        self.assertEqual(
            reg_fs_base.GetValueAsUnsigned(0),
            pthread_self_val.GetValueAsUnsigned(0),
            "fs_base does not equal to pthread_self() value.",
        )

    def test_process_must_be_stopped(self):
        """Check that all register commands error when the process is not stopped."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        pid = self.spawnSubprocess(exe, ["wait_for_attach"]).pid
        # Async so we can enter commands while the process is running.
        self.setAsync(True)
        self.runCmd("process attach --continue -p %d" % pid)

        err_msg = "Command requires a process which is currently stopped."
        self.expect("register read pc", substrs=[err_msg], error=True)
        self.expect("register write pc 0", substrs=[err_msg], error=True)
        self.expect("register info pc", substrs=[err_msg], error=True)
