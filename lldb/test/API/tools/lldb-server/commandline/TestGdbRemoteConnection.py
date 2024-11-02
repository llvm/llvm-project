import gdbremote_testcase
import random
import select
import socket
from lldbsuite.test.decorators import *
from lldbsuite.test.lldbtest import *
from lldbgdbserverutils import Server
import lldbsuite.test.lldbplatformutil

if lldbplatformutil.getHostPlatform() == "windows":
    import ctypes
    import ctypes.wintypes
    from ctypes.wintypes import (BOOL, DWORD, HANDLE, LPCWSTR, LPDWORD, LPVOID)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    PIPE_ACCESS_INBOUND = 1
    FILE_FLAG_FIRST_PIPE_INSTANCE = 0x00080000
    FILE_FLAG_OVERLAPPED = 0x40000000
    PIPE_TYPE_BYTE = 0
    PIPE_REJECT_REMOTE_CLIENTS = 8
    INVALID_HANDLE_VALUE = -1
    ERROR_ACCESS_DENIED = 5
    ERROR_IO_PENDING = 997


    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", LPVOID), ("InternalHigh", LPVOID), ("Offset",
            DWORD), ("OffsetHigh", DWORD), ("hEvent", HANDLE)]

        def __init__(self):
            super(OVERLAPPED, self).__init__(Internal=0, InternalHigh=0,
                Offset=0, OffsetHigh=0, hEvent=None)
    LPOVERLAPPED = ctypes.POINTER(OVERLAPPED)

    CreateNamedPipe = kernel32.CreateNamedPipeW
    CreateNamedPipe.restype = HANDLE
    CreateNamedPipe.argtypes = (LPCWSTR, DWORD, DWORD, DWORD, DWORD, DWORD,
            DWORD, LPVOID)

    ConnectNamedPipe = kernel32.ConnectNamedPipe
    ConnectNamedPipe.restype = BOOL
    ConnectNamedPipe.argtypes = (HANDLE, LPOVERLAPPED)

    CreateEvent = kernel32.CreateEventW
    CreateEvent.restype = HANDLE
    CreateEvent.argtypes = (LPVOID, BOOL, BOOL, LPCWSTR)

    GetOverlappedResultEx = kernel32.GetOverlappedResultEx
    GetOverlappedResultEx.restype = BOOL
    GetOverlappedResultEx.argtypes = (HANDLE, LPOVERLAPPED, LPDWORD, DWORD,
        BOOL)

    ReadFile = kernel32.ReadFile
    ReadFile.restype = BOOL
    ReadFile.argtypes = (HANDLE, LPVOID, DWORD, LPDWORD, LPOVERLAPPED)

    CloseHandle = kernel32.CloseHandle
    CloseHandle.restype = BOOL
    CloseHandle.argtypes = (HANDLE,)

    class Pipe(object):
        def __init__(self, prefix):
            while True:
                self.name = "lldb-" + str(random.randrange(1e10))
                full_name = "\\\\.\\pipe\\" + self.name
                self._handle = CreateNamedPipe(full_name, PIPE_ACCESS_INBOUND |
                        FILE_FLAG_FIRST_PIPE_INSTANCE | FILE_FLAG_OVERLAPPED,
                        PIPE_TYPE_BYTE | PIPE_REJECT_REMOTE_CLIENTS, 1, 4096,
                        4096, 0, None)
                if self._handle != INVALID_HANDLE_VALUE:
                    break
                if ctypes.get_last_error() != ERROR_ACCESS_DENIED:
                    raise ctypes.WinError(ctypes.get_last_error())

            self._overlapped = OVERLAPPED()
            self._overlapped.hEvent = CreateEvent(None, True, False, None)
            result = ConnectNamedPipe(self._handle, self._overlapped)
            assert result == 0
            if ctypes.get_last_error() != ERROR_IO_PENDING:
                raise ctypes.WinError(ctypes.get_last_error())

        def finish_connection(self, timeout):
            if not GetOverlappedResultEx(self._handle, self._overlapped,
                    ctypes.byref(DWORD(0)), timeout*1000, True):
                raise ctypes.WinError(ctypes.get_last_error())

        def read(self, size, timeout):
            buf = ctypes.create_string_buffer(size)
            if not ReadFile(self._handle, ctypes.byref(buf), size, None,
                    self._overlapped):
                if ctypes.get_last_error() != ERROR_IO_PENDING:
                    raise ctypes.WinError(ctypes.get_last_error())
            read = DWORD(0)
            if not GetOverlappedResultEx(self._handle, self._overlapped,
                    ctypes.byref(read), timeout*1000, True):
                raise ctypes.WinError(ctypes.get_last_error())
            return buf.raw[0:read.value]

        def close(self):
            CloseHandle(self._overlapped.hEvent)
            CloseHandle(self._handle)


else:
    class Pipe(object):
        def __init__(self, prefix):
            self.name = os.path.join(prefix, "stub_port_number")
            os.mkfifo(self.name)
            self._fd = os.open(self.name, os.O_RDONLY | os.O_NONBLOCK)

        def finish_connection(self, timeout):
            pass

        def read(self, size, timeout):
            (readers, _, _) = select.select([self._fd], [], [], timeout)
            if self._fd not in readers:
                raise TimeoutError
            return os.read(self._fd, size)

        def close(self):
            os.close(self._fd)


class TestGdbRemoteConnection(gdbremote_testcase.GdbRemoteTestCaseBase):

    @skipIfRemote  # reverse connect is not a supported use case for now
    def test_reverse_connect(self):
        # Reverse connect is the default connection method.
        self.connect_to_debug_monitor()
        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake()

    @skipIfRemote
    def test_named_pipe(self):
        family, type, proto, _, addr = socket.getaddrinfo(
            self.stub_hostname, 0, proto=socket.IPPROTO_TCP)[0]
        self.sock = socket.socket(family, type, proto)
        self.sock.settimeout(self.DEFAULT_TIMEOUT)

        self.addTearDownHook(lambda: self.sock.close())

        pipe = Pipe(self.getBuildDir())

        self.addTearDownHook(lambda: pipe.close())

        args = self.debug_monitor_extra_args
        if lldb.remote_platform:
            args += ["*:0"]
        else:
            args += ["localhost:0"]

        args += ["--named-pipe", pipe.name]

        server = self.spawnSubprocess(
            self.debug_monitor_exe,
            args,
            install_remote=False)

        pipe.finish_connection(self.DEFAULT_TIMEOUT)
        port = pipe.read(10, self.DEFAULT_TIMEOUT)
        # Trim null byte, convert to int
        addr = (addr[0], int(port[:-1]))
        self.sock.connect(addr)
        self._server = Server(self.sock, server)

        # Verify we can do the handshake.  If that works, we'll call it good.
        self.do_handshake()
