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
#include "lldb/Host/windows/ProcessLauncherWindows.h"
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

  assert(m_conpty_handle == INVALID_HANDLE_VALUE &&
         "ConPTY has already been opened");

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

  if (auto error = DrainInitSequences()) {
    Log *log = GetLog(LLDBLog::Host);
    LLDB_LOG_ERROR(log, std::move(error),
                   "failed to finalize ConPTY's setup: {0}");
  }

  return llvm::Error::success();
}

void PseudoConsole::Close() {
  Sleep(50); // FIXME: This mitigates a race condition when closing the
             // PseudoConsole. It's possible that there is still data in the
             // pipe when we try to close it. We should wait until the data has
             // been consumed.
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

llvm::Error PseudoConsole::DrainInitSequences() {
  STARTUPINFOEXW startupinfoex = {};
  startupinfoex.StartupInfo.cb = sizeof(STARTUPINFOEXW);
  startupinfoex.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;

  auto attributelist_or_err = ProcThreadAttributeList::Create(startupinfoex);
  if (!attributelist_or_err)
    return llvm::errorCodeToError(attributelist_or_err.getError());
  ProcThreadAttributeList attributelist = std::move(*attributelist_or_err);
  if (auto error = attributelist.SetupPseudoConsole(m_conpty_handle))
    return error;

  PROCESS_INFORMATION pi = {};

  wchar_t comspec[MAX_PATH];
  DWORD comspecLen = GetEnvironmentVariableW(L"COMSPEC", comspec, MAX_PATH);
  if (comspecLen == 0 || comspecLen >= MAX_PATH)
    return llvm::createStringError(
        std::error_code(GetLastError(), std::system_category()),
        "Failed to get the 'COMSPEC' environment variable");

  std::wstring cmdline_str = std::wstring(comspec) + L" /c 'echo foo && exit'";
  std::vector<wchar_t> cmdline(cmdline_str.begin(), cmdline_str.end());
  cmdline.push_back(L'\0');

  if (!CreateProcessW(/*lpApplicationName=*/comspec, cmdline.data(),
                      /*lpProcessAttributes=*/NULL, /*lpThreadAttributes=*/NULL,
                      /*bInheritHandles=*/TRUE,
                      /*dwCreationFlags=*/EXTENDED_STARTUPINFO_PRESENT |
                          CREATE_UNICODE_ENVIRONMENT,
                      /*lpEnvironment=*/NULL, /*lpCurrentDirectory=*/NULL,
                      /*lpStartupInfo=*/
                      reinterpret_cast<STARTUPINFOW *>(&startupinfoex),
                      /*lpProcessInformation=*/&pi))
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));

  char buf[4096];
  OVERLAPPED ov = {};
  ov.hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

  DWORD read;
  ReadFile(m_conpty_output, buf, sizeof(buf), &read, &ov);

  WaitForSingleObject(pi.hProcess, INFINITE);

  if (GetOverlappedResult(m_conpty_output, &ov, &read, FALSE) && read > 0) {
    ResetEvent(ov.hEvent);
    ReadFile(m_conpty_output, buf, sizeof(buf), &read, &ov);
  }

  CancelIo(m_conpty_output);
  CloseHandle(ov.hEvent);
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  return llvm::Error::success();
}
