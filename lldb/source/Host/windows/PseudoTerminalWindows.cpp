//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PseudoTerminalWindows.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Errno.h"

using namespace lldb_private;

void PseudoTerminalWindows::Close() {
  if (m_conpty_handle != INVALID_HANDLE_VALUE)
    ClosePseudoConsole(m_conpty_handle);
  CloseHandle(m_conpty_input);
  CloseHandle(m_conpty_output);
  m_conpty_handle = INVALID_HANDLE_VALUE;
  m_conpty_input = INVALID_HANDLE_VALUE;
  m_conpty_output = INVALID_HANDLE_VALUE;
}

llvm::Error PseudoTerminalWindows::OpenFirstAvailablePrimary(int oflag) {
  HRESULT hr;
  HANDLE hInputRead = INVALID_HANDLE_VALUE;
  HANDLE hInputWrite = INVALID_HANDLE_VALUE;
  HANDLE hOutputRead = INVALID_HANDLE_VALUE;
  HANDLE hOutputWrite = INVALID_HANDLE_VALUE;

  wchar_t pipe_name[MAX_PATH];
  swprintf(pipe_name, MAX_PATH, L"\\\\.\\pipe\\conpty-%d-%p",
           GetCurrentProcessId(), this);

  hOutputRead =
      CreateNamedPipeW(pipe_name, PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, NULL);
  hOutputWrite = CreateFileW(pipe_name, GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, NULL);

  if (!CreatePipe(&hInputRead, &hInputWrite, NULL, 0))
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));

  COORD consoleSize{256, 25};
  HPCON hPC = INVALID_HANDLE_VALUE;
  hr = CreatePseudoConsole(consoleSize, hInputRead, hOutputWrite, 0, &hPC);
  CloseHandle(hInputRead);
  CloseHandle(hOutputWrite);

  if (FAILED(hr)) {
    CloseHandle(hInputWrite);
    CloseHandle(hOutputRead);
    return llvm::make_error<llvm::StringError>(
        "Failed to create Windows ConPTY pseudo terminal",
        llvm::errc::io_error);
  }

  DWORD mode = PIPE_NOWAIT;
  SetNamedPipeHandleState(hOutputRead, &mode, NULL, NULL);

  m_conpty_handle = hPC;
  m_conpty_output = hOutputRead;
  m_conpty_input = hInputWrite;

  return llvm::Error::success();
}
