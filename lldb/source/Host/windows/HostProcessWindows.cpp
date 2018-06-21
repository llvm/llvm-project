//===-- HostProcessWindows.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/HostProcessWindows.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Utility/FileSpec.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ConvertUTF.h"

#include <psapi.h>

using namespace lldb_private;

namespace {
struct MonitorInfo {
  Host::MonitorChildProcessCallback callback;
  HANDLE process_handle;
};
}

HostProcessWindows::HostProcessWindows()
    : HostNativeProcessBase(), m_owns_handle(true) {}

HostProcessWindows::HostProcessWindows(lldb::process_t process)
    : HostNativeProcessBase(process), m_owns_handle(true) {}

HostProcessWindows::~HostProcessWindows() { Close(); }

void HostProcessWindows::SetOwnsHandle(bool owns) { m_owns_handle = owns; }

Status HostProcessWindows::Terminate() {
  Status error;
  if (m_process == nullptr)
    error.SetError(ERROR_INVALID_HANDLE, lldb::eErrorTypeWin32);

  if (!::TerminateProcess(m_process, 0))
    error.SetError(::GetLastError(), lldb::eErrorTypeWin32);

  return error;
}

Status HostProcessWindows::GetMainModule(FileSpec &file_spec) const {
  Status error;
  if (m_process == nullptr)
    error.SetError(ERROR_INVALID_HANDLE, lldb::eErrorTypeWin32);

  std::vector<wchar_t> wpath(PATH_MAX);
  if (::GetProcessImageFileNameW(m_process, wpath.data(), wpath.size())) {
    std::string path;
    if (llvm::convertWideToUTF8(wpath.data(), path))
      file_spec.SetFile(path, false);
    else
      error.SetErrorString("Error converting path to UTF-8");
  } else
    error.SetError(::GetLastError(), lldb::eErrorTypeWin32);

  return error;
}

lldb::pid_t HostProcessWindows::GetProcessId() const {
  return (m_process == LLDB_INVALID_PROCESS) ? -1 : ::GetProcessId(m_process);
}

bool HostProcessWindows::IsRunning() const {
  if (m_process == nullptr)
    return false;

  DWORD code = 0;
  if (!::GetExitCodeProcess(m_process, &code))
    return false;

  return (code == STILL_ACTIVE);
}

HostThread HostProcessWindows::StartMonitoring(
    const Host::MonitorChildProcessCallback &callback, bool monitor_signals) {
  HostThread monitor_thread;
  MonitorInfo *info = new MonitorInfo;
  info->callback = callback;

  // Since the life of this HostProcessWindows instance and the life of the
  // process may be different, duplicate the handle so that
  // the monitor thread can have ownership over its own copy of the handle.
  HostThread result;
  if (::DuplicateHandle(GetCurrentProcess(), m_process, GetCurrentProcess(),
                        &info->process_handle, 0, FALSE, DUPLICATE_SAME_ACCESS))
    result = ThreadLauncher::LaunchThread("ChildProcessMonitor",
                                          HostProcessWindows::MonitorThread,
                                          info, nullptr);
  return result;
}

lldb::thread_result_t HostProcessWindows::MonitorThread(void *thread_arg) {
  DWORD exit_code;

  MonitorInfo *info = static_cast<MonitorInfo *>(thread_arg);
  if (info) {
    ::WaitForSingleObject(info->process_handle, INFINITE);
    ::GetExitCodeProcess(info->process_handle, &exit_code);
    info->callback(::GetProcessId(info->process_handle), true, 0, exit_code);
    ::CloseHandle(info->process_handle);
    delete (info);
  }
  return 0;
}

void HostProcessWindows::Close() {
  if (m_owns_handle && m_process != LLDB_INVALID_PROCESS)
    ::CloseHandle(m_process);
  m_process = nullptr;
}
