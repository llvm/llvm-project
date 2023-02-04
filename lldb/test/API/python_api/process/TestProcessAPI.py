"""
Test SBProcess APIs, including ReadMemory(), WriteMemory(), and others.
"""

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test.lldbutil import get_stopped_thread, state_type_to_str


class ProcessAPITestCase(TestBase):

    def setUp(self):
        # Call super's setUp().
        TestBase.setUp(self)
        # Find the line number to break inside main().
        self.line = line_number(
            "main.cpp",
            "// Set break point at this line and check variable 'my_char'.")

    def test_scripted_implementation(self):
        self.build()
        exe = self.getBuildArtifact("a.out")

        (target, process, _, _) = \
            lldbutil.run_to_source_breakpoint(self, "Set break point",
                                              lldb.SBFileSpec("main.cpp"))

        self.assertTrue(process, PROCESS_IS_VALID)
        self.assertEqual(process.GetScriptedImplementation(), None)


    def test_read_memory(self):
        """Test Python SBProcess.ReadMemory() API."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        error = lldb.SBError()
        self.assertFalse(val.TypeIsPointerType())
        content = process.ReadMemory(
            val.AddressOf().GetValueAsUnsigned(), 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        if self.TraceOn():
            print("memory content:", content)

        self.expect(
            content,
            "Result from SBProcess.ReadMemory() matches our expected output: 'x'",
            exe=False,
            startstr=b'x')

        # Read (char *)my_char_ptr.
        val = frame.FindValue("my_char_ptr", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)
        cstring = process.ReadCStringFromMemory(
            val.GetValueAsUnsigned(), 256, error)
        if not error.Success():
            self.fail("SBProcess.ReadCStringFromMemory() failed")
        if self.TraceOn():
            print("cstring read is:", cstring)

        self.expect(
            cstring,
            "Result from SBProcess.ReadCStringFromMemory() matches our expected output",
            exe=False,
            startstr='Does it work?')

        # Get the SBValue for the global variable 'my_cstring'.
        val = frame.FindValue("my_cstring", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # Due to the typemap magic (see lldb.swig), we pass in 256 to read at most 256 bytes
        # from the address, and expect to get a Python string as the result
        # object!
        self.assertFalse(val.TypeIsPointerType())
        cstring = process.ReadCStringFromMemory(
            val.AddressOf().GetValueAsUnsigned(), 256, error)
        if not error.Success():
            self.fail("SBProcess.ReadCStringFromMemory() failed")
        if self.TraceOn():
            print("cstring read is:", cstring)

        self.expect(
            cstring,
            "Result from SBProcess.ReadCStringFromMemory() matches our expected output",
            exe=False,
            startstr='lldb.SBProcess.ReadCStringFromMemory() works!')

        # Get the SBValue for the global variable 'my_uint32'.
        val = frame.FindValue("my_uint32", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # Due to the typemap magic (see lldb.swig), we pass in 4 to read 4 bytes
        # from the address, and expect to get an int as the result!
        self.assertFalse(val.TypeIsPointerType())
        my_uint32 = process.ReadUnsignedFromMemory(
            val.AddressOf().GetValueAsUnsigned(), 4, error)
        if not error.Success():
            self.fail("SBProcess.ReadCStringFromMemory() failed")
        if self.TraceOn():
            print("uint32 read is:", my_uint32)

        if my_uint32 != 12345:
            self.fail(
                "Result from SBProcess.ReadUnsignedFromMemory() does not match our expected output")

    def test_write_memory(self):
        """Test Python SBProcess.WriteMemory() API."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_char'.
        val = frame.FindValue("my_char", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # If the variable does not have a load address, there's no sense
        # continuing.
        if not val.GetLocation().startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(), 16)

        # The program logic makes the 'my_char' variable to have memory content as 'x'.
        # But we want to use the WriteMemory() API to assign 'a' to the
        # variable.

        # Now use WriteMemory() API to write 'a' into the global variable.
        error = lldb.SBError()
        result = process.WriteMemory(location, 'a', error)
        if not error.Success() or result != 1:
            self.fail("SBProcess.WriteMemory() failed")

        # Read from the memory location.  This time it should be 'a'.
        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        content = process.ReadMemory(location, 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        if self.TraceOn():
            print("memory content:", content)

        self.expect(
            content,
            "Result from SBProcess.ReadMemory() matches our expected output: 'a'",
            exe=False,
            startstr=b'a')

    def test_access_my_int(self):
        """Test access 'my_int' using Python SBProcess.GetByteOrder() and other APIs."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(
            thread.IsValid(),
            "There should be a thread stopped due to breakpoint")
        frame = thread.GetFrameAtIndex(0)

        # Get the SBValue for the global variable 'my_int'.
        val = frame.FindValue("my_int", lldb.eValueTypeVariableGlobal)
        self.DebugSBValue(val)

        # If the variable does not have a load address, there's no sense
        # continuing.
        if not val.GetLocation().startswith("0x"):
            return

        # OK, let's get the hex location of the variable.
        location = int(val.GetLocation(), 16)

        # Note that the canonical from of the bytearray is little endian.
        from lldbsuite.test.lldbutil import int_to_bytearray, bytearray_to_int

        byteSize = val.GetByteSize()
        bytes = int_to_bytearray(256, byteSize)

        byteOrder = process.GetByteOrder()
        if byteOrder == lldb.eByteOrderBig:
            bytes.reverse()
        elif byteOrder == lldb.eByteOrderLittle:
            pass
        else:
            # Neither big endian nor little endian?  Return for now.
            # Add more logic here if we want to handle other types.
            return

        # The program logic makes the 'my_int' variable to have int type and value of 0.
        # But we want to use the WriteMemory() API to assign 256 to the
        # variable.

        # Now use WriteMemory() API to write 256 into the global variable.
        error = lldb.SBError()
        result = process.WriteMemory(location, bytes, error)
        if not error.Success() or result != byteSize:
            self.fail("SBProcess.WriteMemory() failed")

        # Make sure that the val we got originally updates itself to notice the
        # change:
        self.expect(
            val.GetValue(),
            "SBProcess.ReadMemory() successfully writes (int)256 to the memory location for 'my_int'",
            exe=False,
            startstr='256')

        # And for grins, get the SBValue for the global variable 'my_int'
        # again, to make sure that also tracks the new value:
        val = frame.FindValue("my_int", lldb.eValueTypeVariableGlobal)
        self.expect(
            val.GetValue(),
            "SBProcess.ReadMemory() successfully writes (int)256 to the memory location for 'my_int'",
            exe=False,
            startstr='256')

        # Now read the memory content.  The bytearray should have (byte)1 as
        # the second element.
        content = process.ReadMemory(location, byteSize, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")

        # The bytearray_to_int utility function expects a little endian
        # bytearray.
        if byteOrder == lldb.eByteOrderBig:
            content = bytearray(content, 'ascii')
            content.reverse()

        new_value = bytearray_to_int(content, byteSize)
        if new_value != 256:
            self.fail("Memory content read from 'my_int' does not match (int)256")

        # Dump the memory content....
        if self.TraceOn():
            for i in content:
                print("byte:", i)

    def test_remote_launch(self):
        """Test SBProcess.RemoteLaunch() API with a process not in eStateConnected, and it should fail."""
        self.build()
        exe = self.getBuildArtifact("a.out")

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        if self.TraceOn():
            print("process state:", state_type_to_str(process.GetState()))
        self.assertTrue(process.GetState() != lldb.eStateConnected)

        error = lldb.SBError()
        success = process.RemoteLaunch(
            None, None, None, None, None, None, 0, False, error)
        self.assertTrue(
            not success,
            "RemoteLaunch() should fail for process state != eStateConnected")

    def test_get_num_supported_hardware_watchpoints(self):
        """Test SBProcess.GetNumSupportedHardwareWatchpoints() API with a process."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        breakpoint = target.BreakpointCreateByLocation("main.cpp", self.line)
        self.assertTrue(breakpoint, VALID_BREAKPOINT)

        # Launch the process, and do not stop at the entry point.
        process = target.LaunchSimple(
            None, None, self.get_process_working_directory())

        error = lldb.SBError()
        num = process.GetNumSupportedHardwareWatchpoints(error)
        if self.TraceOn() and error.Success():
            print("Number of supported hardware watchpoints: %d" % num)

    @no_debug_info_test
    @skipIfRemote
    def test_get_process_info(self):
        """Test SBProcess::GetProcessInfo() API with a locally launched process."""
        self.build()
        exe = self.getBuildArtifact("a.out")
        self.runCmd("file " + exe, CURRENT_EXECUTABLE_SET)

        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Launch the process and stop at the entry point.
        launch_info = target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(self.get_process_working_directory())
        launch_flags = launch_info.GetLaunchFlags()
        launch_flags |= lldb.eLaunchFlagStopAtEntry
        launch_info.SetLaunchFlags(launch_flags)
        error = lldb.SBError()
        process = target.Launch(launch_info, error)

        if not error.Success():
            self.fail("Failed to launch process")

        # Verify basic process info can be retrieved successfully
        process_info = process.GetProcessInfo()
        self.assertTrue(process_info.IsValid())
        file_spec = process_info.GetExecutableFile()
        self.assertTrue(file_spec.IsValid())
        process_name = process_info.GetName()
        self.assertIsNotNone(process_name, "Process has a name")
        self.assertGreater(len(process_name), 0, "Process name isn't blank")
        self.assertEqual(file_spec.GetFilename(), "a.out")
        self.assertNotEqual(
            process_info.GetProcessID(), lldb.LLDB_INVALID_PROCESS_ID,
            "Process ID is valid")
        triple = process_info.GetTriple()
        self.assertIsNotNone(triple, "Process has a triple")

        # Additional process info varies by platform, so just check that
        # whatever info was retrieved is consistent and nothing blows up.
        if process_info.UserIDIsValid():
            self.assertNotEqual(
                process_info.GetUserID(), lldb.UINT32_MAX,
                "Process user ID is valid")
        else:
            self.assertEqual(
                process_info.GetUserID(), lldb.UINT32_MAX,
                "Process user ID is invalid")

        if process_info.GroupIDIsValid():
            self.assertNotEqual(
                process_info.GetGroupID(), lldb.UINT32_MAX,
                "Process group ID is valid")
        else:
            self.assertEqual(
                process_info.GetGroupID(), lldb.UINT32_MAX,
                "Process group ID is invalid")

        if process_info.EffectiveUserIDIsValid():
            self.assertNotEqual(
                process_info.GetEffectiveUserID(), lldb.UINT32_MAX,
                "Process effective user ID is valid")
        else:
            self.assertEqual(
                process_info.GetEffectiveUserID(), lldb.UINT32_MAX,
                "Process effective user ID is invalid")

        if process_info.EffectiveGroupIDIsValid():
            self.assertNotEqual(
                process_info.GetEffectiveGroupID(), lldb.UINT32_MAX,
                "Process effective group ID is valid")
        else:
            self.assertEqual(
                process_info.GetEffectiveGroupID(), lldb.UINT32_MAX,
                "Process effective group ID is invalid")

        process_info.GetParentProcessID()

    def test_allocate_deallocate_memory(self):
        """Test Python SBProcess.AllocateMemory() and SBProcess.DeallocateMemory() APIs."""
        self.build()
        (target, process, main_thread, main_breakpoint) = lldbutil.run_to_source_breakpoint(
            self, "// Set break point at this line", lldb.SBFileSpec("main.cpp"))

        # Allocate a block of memory in the target process
        error = lldb.SBError()
        addr = process.AllocateMemory(16384, lldb.ePermissionsReadable, error)
        if not error.Success() or addr == lldb.LLDB_INVALID_ADDRESS:
            self.fail("SBProcess.AllocateMemory() failed")

        # Now use WriteMemory() API to write 'a' into the allocated
        # memory. Note that the debugger can do this even though the
        # block is not set writable.
        result = process.WriteMemory(addr, 'a', error)
        if not error.Success() or result != 1:
            self.fail("SBProcess.WriteMemory() failed")

        # Read from the memory location.  This time it should be 'a'.
        # Due to the typemap magic (see lldb.swig), we pass in 1 to ReadMemory and
        # expect to get a Python string as the result object!
        content = process.ReadMemory(addr, 1, error)
        if not error.Success():
            self.fail("SBProcess.ReadMemory() failed")
        if self.TraceOn():
            print("memory content:", content)

        self.expect(
            content,
            "Result from SBProcess.ReadMemory() matches our expected output: 'a'",
            exe=False,
            startstr=b'a')

        # Verify that the process itself can read the allocated memory
        frame = main_thread.GetFrameAtIndex(0)
        val = frame.EvaluateExpression(
            "test_read(reinterpret_cast<char *>({:#x}))".format(addr))
        self.expect(val.GetValue(),
                    "Result of test_read() matches expected output 'a'",
                    exe=False,
                    startstr="'a'")

        # Verify that the process cannot write into the block
        val = frame.EvaluateExpression(
            "test_write(reinterpret_cast<char *>({:#x}), 'b')".format(addr))
        if val.GetError().Success():
            self.fail(
                "test_write() to allocated memory without write permission unexpectedly succeeded")

        # Deallocate the memory
        error = process.DeallocateMemory(addr)
        if not error.Success():
            self.fail("SBProcess.DeallocateMemory() failed")
