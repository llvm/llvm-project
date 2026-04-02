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

llvm::Error PseudoConsole::CreateOverlappedPipePair(HANDLE &out_read,
                                                    HANDLE &out_write,
                                                    bool inheritable) {
  wchar_t pipe_name[MAX_PATH];
  swprintf(pipe_name, MAX_PATH, L"\\\\.\\pipe\\conpty-lldb-%d-%p",
           GetCurrentProcessId(), this);
  out_read =
      CreateNamedPipeW(pipe_name, PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
                       PIPE_TYPE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, NULL);
  if (out_read == INVALID_HANDLE_VALUE)
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));
  SECURITY_ATTRIBUTES write_sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
  out_write =
      CreateFileW(pipe_name, GENERIC_WRITE, 0, inheritable ? &write_sa : NULL,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (out_write == INVALID_HANDLE_VALUE) {
    CloseHandle(out_read);
    out_read = INVALID_HANDLE_VALUE;
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));
  }

  DWORD mode = PIPE_NOWAIT;
  SetNamedPipeHandleState(out_read, &mode, NULL, NULL);
  return llvm::Error::success();
}

PseudoConsole::~PseudoConsole() {
  Close();
  ClosePseudoConsolePipes();
  CloseAnonymousPipes();
}

llvm::Error PseudoConsole::OpenPseudoConsole() {
  assert(m_mode == Mode::None &&
         "Attempted to open a PseudoConsole in a different mode than None");

  if (!kernel32.IsConPTYAvailable())
    return llvm::make_error<llvm::StringError>("ConPTY is not available",
                                               llvm::errc::io_error);

  assert(m_conpty_handle == INVALID_HANDLE_VALUE &&
         "ConPTY has already been opened");

  // A 4096 bytes buffer should be large enough for the majority of console
  // burst outputs.
  wchar_t pipe_name[MAX_PATH];
  swprintf(pipe_name, MAX_PATH, L"\\\\.\\pipe\\conpty-lldb-%d-%p",
           GetCurrentProcessId(), this);
  HANDLE hOutputRead = INVALID_HANDLE_VALUE;
  HANDLE hOutputWrite = INVALID_HANDLE_VALUE;
  if (auto err = CreateOverlappedPipePair(hOutputRead, hOutputWrite, false))
    return err;

  HANDLE hInputRead = INVALID_HANDLE_VALUE;
  HANDLE hInputWrite = INVALID_HANDLE_VALUE;
  if (!CreatePipe(&hInputRead, &hInputWrite, NULL, 0)) {
    CloseHandle(hOutputRead);
    CloseHandle(hOutputWrite);
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));
  }

  COORD consoleSize{80, 25};
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi))
    consoleSize = {
        static_cast<SHORT>(csbi.srWindow.Right - csbi.srWindow.Left + 1),
        static_cast<SHORT>(csbi.srWindow.Bottom - csbi.srWindow.Top + 1)};
  HPCON hPC = INVALID_HANDLE_VALUE;
  HRESULT hr = kernel32.CreatePseudoConsole(consoleSize, hInputRead,
                                            hOutputWrite, 0, &hPC);
  CloseHandle(hInputRead);
  CloseHandle(hOutputWrite);

  if (FAILED(hr)) {
    CloseHandle(hInputWrite);
    CloseHandle(hOutputRead);
    return llvm::make_error<llvm::StringError>(
        "Failed to create Windows ConPTY pseudo terminal",
        llvm::errc::io_error);
  }

  m_conpty_handle = hPC;
  m_conpty_output = hOutputRead;
  m_conpty_input = hInputWrite;
  m_mode = Mode::ConPTY;

  if (auto error = DrainInitSequences()) {
    Log *log = GetLog(LLDBLog::Host);
    LLDB_LOG_ERROR(log, std::move(error),
                   "failed to finalize ConPTY's setup: {0}");
  }

  return llvm::Error::success();
}

bool PseudoConsole::IsConnected() const {
  if (m_mode == Mode::Pipe)
    return m_conpty_input != INVALID_HANDLE_VALUE &&
           m_conpty_output != INVALID_HANDLE_VALUE;
  return m_conpty_handle != INVALID_HANDLE_VALUE &&
         m_conpty_input != INVALID_HANDLE_VALUE &&
         m_conpty_output != INVALID_HANDLE_VALUE;
}

void PseudoConsole::Close() {
  SetStopping(true);
  std::unique_lock<std::mutex> guard(m_mutex);
  if (m_conpty_handle != INVALID_HANDLE_VALUE)
    kernel32.ClosePseudoConsole(m_conpty_handle);
  m_conpty_handle = INVALID_HANDLE_VALUE;
  SetStopping(false);
  m_cv.notify_all();
}

void PseudoConsole::ClosePseudoConsolePipes() {
  if (m_conpty_input != INVALID_HANDLE_VALUE)
    CloseHandle(m_conpty_input);
  if (m_conpty_output != INVALID_HANDLE_VALUE)
    CloseHandle(m_conpty_output);

  m_conpty_input = INVALID_HANDLE_VALUE;
  m_conpty_output = INVALID_HANDLE_VALUE;
}

void PseudoConsole::CloseAnonymousPipes() {
  if (m_pipe_child_stdin != INVALID_HANDLE_VALUE)
    CloseHandle(m_pipe_child_stdin);
  if (m_pipe_child_stdout != INVALID_HANDLE_VALUE)
    CloseHandle(m_pipe_child_stdout);

  m_pipe_child_stdin = INVALID_HANDLE_VALUE;
  m_pipe_child_stdout = INVALID_HANDLE_VALUE;
}

llvm::Error PseudoConsole::OpenAnonymousPipes() {
  assert(m_mode == Mode::None &&
         "Attempted to open a AnonymousPipes in a different mode than None");

  SECURITY_ATTRIBUTES sa = {sizeof(SECURITY_ATTRIBUTES), NULL, TRUE};
  HANDLE hStdinRead = INVALID_HANDLE_VALUE;
  HANDLE hStdinWrite = INVALID_HANDLE_VALUE;
  if (!CreatePipe(&hStdinRead, &hStdinWrite, &sa, 0))
    return llvm::errorCodeToError(
        std::error_code(GetLastError(), std::system_category()));
  // Parent write end must not be inherited by the child.
  SetHandleInformation(hStdinWrite, HANDLE_FLAG_INHERIT, 0);

  HANDLE hStdoutRead = INVALID_HANDLE_VALUE;
  HANDLE hStdoutWrite = INVALID_HANDLE_VALUE;
  if (auto err = CreateOverlappedPipePair(hStdoutRead, hStdoutWrite, true)) {
    CloseHandle(hStdinRead);
    CloseHandle(hStdinWrite);
    return err;
  }

  m_conpty_input = hStdinWrite;
  m_conpty_output = hStdoutRead;
  m_pipe_child_stdin = hStdinRead;
  m_pipe_child_stdout = hStdoutWrite;
  m_mode = Mode::Pipe;
  return llvm::Error::success();
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

  std::wstring cmdline_str = std::wstring(comspec) + L" /c cls";
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
