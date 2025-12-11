//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PseudoConsole.h"

#include <mutex>

#include "lldb/Host/windows/windows.h"

#include "llvm/Support/Errc.h"
#include "llvm/Support/Errno.h"

using namespace lldb_private;

typedef HRESULT(WINAPI *CreatePseudoConsole_t)(COORD size, HANDLE hInput,
                                               HANDLE hOutput, DWORD dwFlags,
                                               HPCON *phPC);

typedef HRESULT(WINAPI *ResizePseudoConsole_t)(HPCON hPC, COORD size);

typedef VOID(WINAPI *ClosePseudoConsole_t)(HPCON hPC);

class ConPTY {
public:
  static bool Initialize() {
    std::lock_guard<std::mutex> guard(m_initialized_mutex);

    if (!m_initialized) {
      m_initialized = true;

      HMODULE hMod = LoadLibraryW(L"kernel32.dll");
      if (!hMod) {
        return false;
      }

      pCreate =
          (CreatePseudoConsole_t)GetProcAddress(hMod, "CreatePseudoConsole");
      pClose = (ClosePseudoConsole_t)GetProcAddress(hMod, "ClosePseudoConsole");

      m_success = (pCreate && pClose);
    }

    return m_success;
  }

  static bool IsAvailable() { return Initialize(); }

  static CreatePseudoConsole_t Create() {
    Initialize();
    return pCreate;
  }

  static ClosePseudoConsole_t Close() {
    Initialize();
    return pClose;
  }

private:
  static CreatePseudoConsole_t pCreate;
  static ClosePseudoConsole_t pClose;
  static std::mutex m_initialized_mutex;
  static bool m_initialized;
  static bool m_success;
};

CreatePseudoConsole_t ConPTY::pCreate = nullptr;
ClosePseudoConsole_t ConPTY::pClose = nullptr;
std::mutex ConPTY::m_initialized_mutex{};
bool ConPTY::m_initialized = false;
bool ConPTY::m_success = false;

llvm::Error PseudoConsole::OpenPseudoConsole() {
  if (!ConPTY::IsAvailable())
    return llvm::make_error<llvm::StringError>("ConPTY is not available",
                                               llvm::errc::io_error);
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

  COORD consoleSize{256, 25};
  HPCON hPC = INVALID_HANDLE_VALUE;
  hr = ConPTY::Create()(consoleSize, hInputRead, hOutputWrite, 0, &hPC);
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
    ConPTY::Close()(m_conpty_handle);
  CloseHandle(m_conpty_input);
  CloseHandle(m_conpty_output);
  m_conpty_handle = INVALID_HANDLE_VALUE;
  m_conpty_input = INVALID_HANDLE_VALUE;
  m_conpty_output = INVALID_HANDLE_VALUE;
}