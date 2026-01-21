//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PseudoConsole.h"

#include <mutex>

#include "lldb/Host/windows/PipeWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/LLDBLog.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Errno.h"

using namespace lldb_private;

typedef HRESULT(WINAPI *CreatePseudoConsole_t)(COORD size, HANDLE hInput,
                                               HANDLE hOutput, DWORD dwFlags,
                                               HPCON *phPC);

typedef VOID(WINAPI *ClosePseudoConsole_t)(HPCON hPC);

struct Kernel32 {
  Kernel32() {
    hModule = LoadLibraryW(L"kernel32.dll");
    if (!hModule) {
      llvm::Error err = llvm::errorCodeToError(
          std::error_code(GetLastError(), std::system_category()));
      LLDB_LOG_ERROR(GetLog(LLDBLog::Host), std::move(err),
                     "Could not load kernel32: {0}");
      return;
    }
    CreatePseudoConsole_ =
        (CreatePseudoConsole_t)GetProcAddress(hModule, "CreatePseudoConsole");
    ClosePseudoConsole_ =
        (ClosePseudoConsole_t)GetProcAddress(hModule, "ClosePseudoConsole");
    isAvailable = (CreatePseudoConsole_ && ClosePseudoConsole_);
  }

  HRESULT CreatePseudoConsole(COORD size, HANDLE hInput, HANDLE hOutput,
                              DWORD dwFlags, HPCON *phPC) {
    assert(CreatePseudoConsole_ && "CreatePseudoConsole is not available!");
    return CreatePseudoConsole_(size, hInput, hOutput, dwFlags, phPC);
  }

  VOID ClosePseudoConsole(HPCON hPC) {
    assert(ClosePseudoConsole_ && "ClosePseudoConsole is not available!");
    return ClosePseudoConsole_(hPC);
  }

  bool IsConPTYAvailable() { return isAvailable; }

private:
  HMODULE hModule;
  CreatePseudoConsole_t CreatePseudoConsole_;
  ClosePseudoConsole_t ClosePseudoConsole_;
  bool isAvailable;
};

static Kernel32 kernel32;

PseudoConsole::~PseudoConsole() { Close(); }

llvm::Error PseudoConsole::OpenPseudoConsole() {
  if (!kernel32.IsConPTYAvailable())
    return llvm::make_error<llvm::StringError>("ConPTY is not available",
                                               llvm::errc::io_error);

  // close any previously opened handles
  Close();

  HRESULT hr;
  HANDLE hInputRead = INVALID_HANDLE_VALUE;
  HANDLE hInputWrite = INVALID_HANDLE_VALUE;
  HANDLE hOutputRead = INVALID_HANDLE_VALUE;
  HANDLE hOutputWrite = INVALID_HANDLE_VALUE;

  wchar_t pipe_name[MAX_PATH];
  swprintf(pipe_name, MAX_PATH, L"\\\\.\\pipe\\conpty-lldb-%d-%p",
           GetCurrentProcessId(), this);

  // A 4096 bytes buffer should be large enough for the majority of console
  // burst outputs.
  hOutputRead =
      CreateNamedPipeW(pipe_name, PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, NULL);
  hOutputWrite = CreateFileW(pipe_name, GENERIC_WRITE, 0, NULL, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL, NULL);

  if (!CreatePipe(&hInputRead, &hInputWrite, NULL, 0))
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));

  COORD consoleSize{80, 25};
  HPCON hPC = INVALID_HANDLE_VALUE;
  hr = kernel32.CreatePseudoConsole(consoleSize, hInputRead, hOutputWrite, 0,
                                    &hPC);
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

void PseudoConsole::Close() {
  if (m_conpty_handle != INVALID_HANDLE_VALUE)
    kernel32.ClosePseudoConsole(m_conpty_handle);
  if (m_conpty_input != INVALID_HANDLE_VALUE)
    CloseHandle(m_conpty_input);
  if (m_conpty_output != INVALID_HANDLE_VALUE)
    CloseHandle(m_conpty_output);

  m_conpty_handle = INVALID_HANDLE_VALUE;
  m_conpty_input = INVALID_HANDLE_VALUE;
  m_conpty_output = INVALID_HANDLE_VALUE;
}
