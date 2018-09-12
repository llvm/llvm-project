//===-- ProcessLauncherWindows.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/ProcessLauncherWindows.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Target/ProcessLaunchInfo.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ConvertUTF.h"

#include <string>
#include <vector>

using namespace lldb;
using namespace lldb_private;

namespace {
void CreateEnvironmentBuffer(const Environment &env,
                             std::vector<char> &buffer) {
  if (env.size() == 0)
    return;

  // Environment buffer is a null terminated list of null terminated strings
  for (const auto &KV : env) {
    std::wstring warg;
    if (llvm::ConvertUTF8toWide(Environment::compose(KV), warg)) {
      buffer.insert(buffer.end(), (char *)warg.c_str(),
                    (char *)(warg.c_str() + warg.size() + 1));
    }
  }
  // One null wchar_t (to end the block) is two null bytes
  buffer.push_back(0);
  buffer.push_back(0);
}
}

HostProcess
ProcessLauncherWindows::LaunchProcess(const ProcessLaunchInfo &launch_info,
                                      Status &error) {
  error.Clear();

  std::string executable;
  std::string commandLine;
  std::vector<char> environment;
  STARTUPINFO startupinfo = {};
  PROCESS_INFORMATION pi = {};

  HANDLE stdin_handle = GetStdioHandle(launch_info, STDIN_FILENO);
  HANDLE stdout_handle = GetStdioHandle(launch_info, STDOUT_FILENO);
  HANDLE stderr_handle = GetStdioHandle(launch_info, STDERR_FILENO);

  startupinfo.cb = sizeof(startupinfo);
  startupinfo.dwFlags |= STARTF_USESTDHANDLES;
  startupinfo.hStdError =
      stderr_handle ? stderr_handle : ::GetStdHandle(STD_ERROR_HANDLE);
  startupinfo.hStdInput =
      stdin_handle ? stdin_handle : ::GetStdHandle(STD_INPUT_HANDLE);
  startupinfo.hStdOutput =
      stdout_handle ? stdout_handle : ::GetStdHandle(STD_OUTPUT_HANDLE);

  const char *hide_console_var =
      getenv("LLDB_LAUNCH_INFERIORS_WITHOUT_CONSOLE");
  if (hide_console_var &&
      llvm::StringRef(hide_console_var).equals_lower("true")) {
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW;
    startupinfo.wShowWindow = SW_HIDE;
  }

  DWORD flags = CREATE_NEW_CONSOLE | CREATE_UNICODE_ENVIRONMENT;
  if (launch_info.GetFlags().Test(eLaunchFlagDebug))
    flags |= DEBUG_ONLY_THIS_PROCESS;

  if (launch_info.GetFlags().Test(eLaunchFlagDisableSTDIO))
    flags &= ~CREATE_NEW_CONSOLE;

  LPVOID env_block = nullptr;
  ::CreateEnvironmentBuffer(launch_info.GetEnvironment(), environment);
  if (!environment.empty())
    env_block = environment.data();

  executable = launch_info.GetExecutableFile().GetPath();
  launch_info.GetArguments().GetQuotedCommandString(commandLine);

  std::wstring wexecutable, wcommandLine, wworkingDirectory;
  llvm::ConvertUTF8toWide(executable, wexecutable);
  llvm::ConvertUTF8toWide(commandLine, wcommandLine);
  llvm::ConvertUTF8toWide(launch_info.GetWorkingDirectory().GetCString(),
                          wworkingDirectory);

  wcommandLine.resize(PATH_MAX); // Needs to be over-allocated because
                                 // CreateProcessW can modify it
  BOOL result = ::CreateProcessW(
      wexecutable.c_str(), &wcommandLine[0], NULL, NULL, TRUE, flags, env_block,
      wworkingDirectory.size() == 0 ? NULL : wworkingDirectory.c_str(),
      &startupinfo, &pi);

  if (!result) {
    // Call GetLastError before we make any other system calls.
    error.SetError(::GetLastError(), eErrorTypeWin32);
  }

  if (result) {
    // Do not call CloseHandle on pi.hProcess, since we want to pass that back
    // through the HostProcess.
    ::CloseHandle(pi.hThread);
  }

  if (stdin_handle)
    ::CloseHandle(stdin_handle);
  if (stdout_handle)
    ::CloseHandle(stdout_handle);
  if (stderr_handle)
    ::CloseHandle(stderr_handle);

  if (!result)
    return HostProcess();

  return HostProcess(pi.hProcess);
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

  llvm::StringRef path = action->GetPath();
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

  std::wstring wpath;
  llvm::ConvertUTF8toWide(path, wpath);
  HANDLE result = ::CreateFileW(wpath.c_str(), access, share, &secattr, create,
                                flags, NULL);
  return (result == INVALID_HANDLE_VALUE) ? NULL : result;
}
