//===-- ProcessLauncherWindows.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/windows/PseudoConsole.h"
#include "lldb/Host/windows/windows.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WindowsError.h"

#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;

/// Create a UTF-16 environment block to use with CreateProcessW.
///
/// The buffer is a sequence of null-terminated UTF-16 strings, followed by an
/// extra L'\0' (two bytes of 0). An empty environment must have one
/// empty string, followed by an extra L'\0'.
///
/// The keys are sorted to comply with the CreateProcess API calling convention.
///
/// Ensure that the resulting buffer is used in conjunction with
/// CreateProcessW and be sure that dwCreationFlags includes
/// CREATE_UNICODE_ENVIRONMENT.
///
/// \param env The Environment object to convert.
/// \returns The sorted sequence of environment variables and their values,
/// separated by null terminators. The vector is guaranteed to never be empty.
static std::vector<wchar_t> CreateEnvironmentBufferW(const Environment &env) {
  std::vector<std::wstring> env_entries;
  for (const auto &KV : env) {
    std::wstring wentry;
    if (llvm::ConvertUTF8toWide(Environment::compose(KV), wentry))
      env_entries.push_back(std::move(wentry));
  }
  std::sort(env_entries.begin(), env_entries.end(),
            [](const std::wstring &a, const std::wstring &b) {
              return _wcsicmp(a.c_str(), b.c_str()) < 0;
            });

  std::vector<wchar_t> buffer;
  for (const auto &env_entry : env_entries) {
    buffer.insert(buffer.end(), env_entry.begin(), env_entry.end());
    buffer.push_back(L'\0');
  }

  if (buffer.empty())
    buffer.push_back(L'\0'); // If there are no environment variables, we have
                             // to ensure there are 4 zero bytes in the buffer.
  buffer.push_back(L'\0');

  return buffer;
}

/// Flattens an Args object into a Windows command-line wide string.
///
/// Returns an empty string if args is empty.
///
/// \param args The Args object to flatten.
/// \returns A wide string containing the flattened command line.
static llvm::ErrorOr<std::wstring>
GetFlattenedWindowsCommandStringW(Args args) {
  if (args.empty())
    return L"";

  std::vector<llvm::StringRef> args_ref;
  for (auto &entry : args.entries())
    args_ref.push_back(entry.ref());

  return llvm::sys::flattenWindowsCommandLine(args_ref);
}

llvm::ErrorOr<ProcThreadAttributeList>
ProcThreadAttributeList::Create(STARTUPINFOEXW &startupinfoex) {
  SIZE_T attributelist_size = 0;
  InitializeProcThreadAttributeList(/*lpAttributeList=*/nullptr,
                                    /*dwAttributeCount=*/1, /*dwFlags=*/0,
                                    &attributelist_size);

  startupinfoex.lpAttributeList =
      static_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(malloc(attributelist_size));

  if (!startupinfoex.lpAttributeList)
    return llvm::mapWindowsError(ERROR_OUTOFMEMORY);

  if (!InitializeProcThreadAttributeList(startupinfoex.lpAttributeList,
                                         /*dwAttributeCount=*/1,
                                         /*dwFlags=*/0, &attributelist_size)) {
    free(startupinfoex.lpAttributeList);
    return llvm::mapWindowsError(GetLastError());
  }

  return ProcThreadAttributeList(startupinfoex.lpAttributeList);
}

HostProcess
ProcessLauncherWindows::LaunchProcess(const ProcessLaunchInfo &launch_info,
                                      Status &error) {
  error.Clear();

  STARTUPINFOEXW startupinfoex = {};
  startupinfoex.StartupInfo.cb = sizeof(STARTUPINFOEXW);
  startupinfoex.StartupInfo.dwFlags |= STARTF_USESTDHANDLES;

  HPCON hPC = launch_info.GetPTY().GetPseudoTerminalHandle();
  bool use_pty = hPC != INVALID_HANDLE_VALUE &&
                 launch_info.GetNumFileActions() == 0 &&
                 launch_info.GetFlags().Test(lldb::eLaunchFlagLaunchInTTY);

  HANDLE stdin_handle = GetStdioHandle(launch_info, STDIN_FILENO);
  HANDLE stdout_handle = GetStdioHandle(launch_info, STDOUT_FILENO);
  HANDLE stderr_handle = GetStdioHandle(launch_info, STDERR_FILENO);
  llvm::scope_exit close_handles([&] {
    if (stdin_handle)
      ::CloseHandle(stdin_handle);
    if (stdout_handle)
      ::CloseHandle(stdout_handle);
    if (stderr_handle)
      ::CloseHandle(stderr_handle);
  });

  auto attributelist_or_err = ProcThreadAttributeList::Create(startupinfoex);
  if (!attributelist_or_err) {
    error = attributelist_or_err.getError();
    return HostProcess();
  }
  llvm::scope_exit delete_attributelist(
      [&] { DeleteProcThreadAttributeList(startupinfoex.lpAttributeList); });

  std::vector<HANDLE> inherited_handles;
  if (use_pty) {
    if (!UpdateProcThreadAttribute(startupinfoex.lpAttributeList, 0,
                                   PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE, hPC,
                                   sizeof(hPC), NULL, NULL)) {
      error = Status(::GetLastError(), eErrorTypeWin32);
      return HostProcess();
    }
  } else {
    auto inherited_handles_or_err = GetInheritedHandles(
        launch_info, startupinfoex, stdout_handle, stderr_handle, stdin_handle);
    if (!inherited_handles_or_err) {
      error = Status(inherited_handles_or_err.getError());
      return HostProcess();
    }
    inherited_handles = std::move(*inherited_handles_or_err);
  }

  const char *hide_console_var =
      getenv("LLDB_LAUNCH_INFERIORS_WITHOUT_CONSOLE");
  if (hide_console_var &&
      llvm::StringRef(hide_console_var).equals_insensitive("true")) {
    startupinfoex.StartupInfo.dwFlags |= STARTF_USESHOWWINDOW;
    startupinfoex.StartupInfo.wShowWindow = SW_HIDE;
  }

  DWORD flags = CREATE_NEW_CONSOLE | CREATE_UNICODE_ENVIRONMENT |
                EXTENDED_STARTUPINFO_PRESENT;
  if (launch_info.GetFlags().Test(eLaunchFlagDebug))
    flags |= DEBUG_ONLY_THIS_PROCESS;

  if (launch_info.GetFlags().Test(eLaunchFlagDisableSTDIO) || use_pty)
    flags &= ~CREATE_NEW_CONSOLE;

  std::vector<wchar_t> environment =
      CreateEnvironmentBufferW(launch_info.GetEnvironment());

  auto wcommandLineOrErr =
      GetFlattenedWindowsCommandStringW(launch_info.GetArguments());
  if (!wcommandLineOrErr) {
    error = Status(wcommandLineOrErr.getError());
    return HostProcess();
  }
  std::wstring wcommandLine = *wcommandLineOrErr;
  // If the command line is empty, it's best to pass a null pointer to tell
  // CreateProcessW to use the executable name as the command line.  If the
  // command line is not empty, its contents may be modified by CreateProcessW.
  WCHAR *pwcommandLine = wcommandLine.empty() ? nullptr : &wcommandLine[0];

  std::wstring wexecutable, wworkingDirectory;
  llvm::ConvertUTF8toWide(launch_info.GetExecutableFile().GetPath(),
                          wexecutable);
  llvm::ConvertUTF8toWide(launch_info.GetWorkingDirectory().GetPath(),
                          wworkingDirectory);

  PROCESS_INFORMATION pi = {};

  BOOL result = ::CreateProcessW(
      wexecutable.c_str(), pwcommandLine, NULL, NULL,
      /*bInheritHandles=*/!inherited_handles.empty(), flags, environment.data(),
      wworkingDirectory.size() == 0 ? NULL : wworkingDirectory.c_str(),
      reinterpret_cast<STARTUPINFOW *>(&startupinfoex), &pi);

  if (!result) {
    // Call GetLastError before we make any other system calls.
    error = Status(::GetLastError(), eErrorTypeWin32);
    // Note that error 50 ("The request is not supported") will occur if you
    // try debug a 64-bit inferior from a 32-bit LLDB.
  }

  if (result) {
    // Do not call CloseHandle on pi.hProcess, since we want to pass that back
    // through the HostProcess.
    ::CloseHandle(pi.hThread);
  }

  if (!result)
    return HostProcess();

  return HostProcess(pi.hProcess);
}

llvm::ErrorOr<std::vector<HANDLE>> ProcessLauncherWindows::GetInheritedHandles(
    const ProcessLaunchInfo &launch_info, STARTUPINFOEXW &startupinfoex,
    HANDLE stdout_handle, HANDLE stderr_handle, HANDLE stdin_handle) {
  std::vector<HANDLE> inherited_handles;

  startupinfoex.StartupInfo.hStdError =
      stderr_handle ? stderr_handle : GetStdHandle(STD_ERROR_HANDLE);
  startupinfoex.StartupInfo.hStdInput =
      stdin_handle ? stdin_handle : GetStdHandle(STD_INPUT_HANDLE);
  startupinfoex.StartupInfo.hStdOutput =
      stdout_handle ? stdout_handle : GetStdHandle(STD_OUTPUT_HANDLE);

  if (startupinfoex.StartupInfo.hStdError)
    inherited_handles.push_back(startupinfoex.StartupInfo.hStdError);
  if (startupinfoex.StartupInfo.hStdInput)
    inherited_handles.push_back(startupinfoex.StartupInfo.hStdInput);
  if (startupinfoex.StartupInfo.hStdOutput)
    inherited_handles.push_back(startupinfoex.StartupInfo.hStdOutput);

  for (size_t i = 0; i < launch_info.GetNumFileActions(); ++i) {
    const FileAction *act = launch_info.GetFileActionAtIndex(i);
    if (act->GetAction() == FileAction::eFileActionDuplicate &&
        act->GetFD() == act->GetActionArgument())
      inherited_handles.push_back(reinterpret_cast<HANDLE>(act->GetFD()));
  }

  if (inherited_handles.empty())
    return inherited_handles;

  if (!UpdateProcThreadAttribute(
          startupinfoex.lpAttributeList, /*dwFlags=*/0,
          PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_handles.data(),
          inherited_handles.size() * sizeof(HANDLE),
          /*lpPreviousValue=*/nullptr, /*lpReturnSize=*/nullptr))
    return llvm::mapWindowsError(::GetLastError());

  return inherited_handles;
}

HANDLE
ProcessLauncherWindows::GetStdioHandle(const ProcessLaunchInfo &launch_info,
                                       int fd) {
  const FileAction *action = launch_info.GetFileActionForFD(fd);
  if (action == nullptr)
    return NULL;
  SECURITY_ATTRIBUTES secattr = {};
  secattr.nLength = sizeof(SECURITY_ATTRIBUTES);
  secattr.bInheritHandle = TRUE;

  DWORD access = 0;
  DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
  DWORD create = 0;
  DWORD flags = 0;
  if (fd == STDIN_FILENO) {
    access = GENERIC_READ;
    create = OPEN_EXISTING;
    flags = FILE_ATTRIBUTE_READONLY;
  }
  if (fd == STDOUT_FILENO || fd == STDERR_FILENO) {
    access = GENERIC_WRITE;
    create = CREATE_ALWAYS;
    if (fd == STDERR_FILENO)
      flags = FILE_FLAG_WRITE_THROUGH;
  }

  const std::string path = action->GetFileSpec().GetPath();
  std::wstring wpath;
  llvm::ConvertUTF8toWide(path, wpath);
  HANDLE result = ::CreateFileW(wpath.c_str(), access, share, &secattr, create,
                                flags, NULL);
  return (result == INVALID_HANDLE_VALUE) ? NULL : result;
}
