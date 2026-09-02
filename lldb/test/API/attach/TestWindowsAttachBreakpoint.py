"""
Test that lldb ignores the Windows loader breakpoint when attaching to a
process, but still stops at a genuine breakpoint instruction (an int3) that
lives in the program's own code.
"""

import ctypes

import lldb
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbsuite.test import lldbutil


class WindowsAttachLoaderBreakpointTestCase(TestBase):
    def create_suspended_process(self, exe):
        """Create ``exe`` suspended and return its (pid, hProcess, hThread).

        This mirrors what the lldb-dap runInTerminal launcher does on Windows:
        the debuggee is started with CREATE_SUSPENDED so lldb can attach while
        the process is still being initialized by the loader. The launcher then
        resumes the main thread once the debugger has attached.
        """
        from ctypes import wintypes

        CREATE_SUSPENDED = 0x00000004

        class STARTUPINFOW(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("lpReserved", wintypes.LPWSTR),
                ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR),
                ("dwX", wintypes.DWORD),
                ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD),
                ("dwYSize", wintypes.DWORD),
                ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD),
                ("dwFillAttribute", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD),
                ("cbReserved2", wintypes.WORD),
                ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
                ("hStdInput", wintypes.HANDLE),
                ("hStdOutput", wintypes.HANDLE),
                ("hStdError", wintypes.HANDLE),
            ]

        class PROCESS_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("hProcess", wintypes.HANDLE),
                ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD),
                ("dwThreadId", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            ctypes.POINTER(STARTUPINFOW),
            ctypes.POINTER(PROCESS_INFORMATION),
        ]
        kernel32.CreateProcessW.restype = wintypes.BOOL

        startupinfo = STARTUPINFOW()
        startupinfo.cb = ctypes.sizeof(STARTUPINFOW)
        process_information = PROCESS_INFORMATION()

        # CreateProcessW may modify the command-line buffer, so it must be
        # writable.
        command_line = ctypes.create_unicode_buffer('"{}"'.format(exe))

        if not kernel32.CreateProcessW(
            exe,
            command_line,
            None,
            None,
            False,
            CREATE_SUSPENDED,
            None,
            None,
            ctypes.byref(startupinfo),
            ctypes.byref(process_information),
        ):
            raise OSError(ctypes.get_last_error(), "CreateProcessW failed")

        return (
            process_information.dwProcessId,
            process_information.hProcess,
            process_information.hThread,
        )

    @skipUnlessWindows
    def test_attach_ignores_loader_breakpoint(self):
        """
        lldb must not report the loader's int3 (raised in a system module while
        attaching) as a user-visible stop, but must still stop at the
        __builtin_debugtrap() in the program's own code.
        """
        self.build()
        exe = self.getBuildArtifact("a.out")

        pid, hProcess, hThread = self.create_suspended_process(exe)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.ResumeThread.argtypes = [ctypes.c_void_p]
        kernel32.ResumeThread.restype = ctypes.c_uint

        def cleanup():
            kernel32.TerminateProcess(hProcess, 0)
            kernel32.CloseHandle(hThread)
            kernel32.CloseHandle(hProcess)

        self.addTearDownHook(cleanup)

        self.dbg.SetAsync(False)

        # Attach while the process is suspended and still being initialized.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        error = lldb.SBError()
        process = target.AttachToProcessWithID(self.dbg.GetListener(), pid, error)
        self.assertSuccess(error, "attach to the suspended process")
        self.assertState(process.GetState(), lldb.eStateStopped)

        self.assertNotEqual(
            kernel32.ResumeThread(hThread), 0xFFFFFFFF, "ResumeThread failed"
        )

        # Continuing must run past the loader breakpoint (skipped because it is
        # raised in a system module) and stop at the program's own
        # __builtin_debugtrap().
        process.Continue()
        self.assertState(process.GetState(), lldb.eStateStopped)

        thread = lldbutil.get_stopped_thread(process, lldb.eStopReasonException)
        self.assertIsNotNone(
            thread, "the process should stop at the user __builtin_debugtrap()"
        )
        function_name = thread.GetFrameAtIndex(0).GetFunctionName()
        self.assertIn(
            "main",
            function_name,
            "expected to stop at the program's __builtin_debugtrap(), but "
            "stopped in '{}' (a spurious loader breakpoint in a system module "
            "was not skipped)".format(function_name),
        )

        process.Continue()
        self.assertState(process.GetState(), lldb.eStateExited)
        self.assertEqual(process.GetExitStatus(), 0)
