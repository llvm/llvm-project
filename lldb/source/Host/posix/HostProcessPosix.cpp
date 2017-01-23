//===-- HostProcessPosix.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/HostProcessPosix.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"

#include "llvm/ADT/STLExtras.h"

#include <limits.h>

using namespace lldb_private;

namespace {
const int kInvalidPosixProcess = 0;
}

HostProcessPosix::HostProcessPosix()
    : HostNativeProcessBase(kInvalidPosixProcess) {}

HostProcessPosix::HostProcessPosix(lldb::process_t process)
    : HostNativeProcessBase(process) {}

HostProcessPosix::~HostProcessPosix() {}

Error HostProcessPosix::Signal(int signo) const {
  if (m_process == kInvalidPosixProcess) {
    Error error;
    error.SetErrorString("HostProcessPosix refers to an invalid process");
    return error;
  }

  return HostProcessPosix::Signal(m_process, signo);
}

Error HostProcessPosix::Signal(lldb::process_t process, int signo) {
  Error error;

  if (-1 == ::kill(process, signo))
    error.SetErrorToErrno();

  return error;
}

Error HostProcessPosix::Terminate() { return Signal(SIGKILL); }

Error HostProcessPosix::GetMainModule(FileSpec &file_spec) const {
  Error error;

  // Use special code here because proc/[pid]/exe is a symbolic link.
  char link_path[PATH_MAX];
  if (snprintf(link_path, PATH_MAX, "/proc/%" PRIu64 "/exe", m_process) != 1) {
    error.SetErrorString("Unable to build /proc/<pid>/exe string");
    return error;
  }

  error = FileSystem::Readlink(FileSpec{link_path, false}, file_spec);
  if (!error.Success())
    return error;

  // If the binary has been deleted, the link name has " (deleted)" appended.
  // Remove if there.
  if (file_spec.GetFilename().GetStringRef().endswith(" (deleted)")) {
    const char *filename = file_spec.GetFilename().GetCString();
    static const size_t deleted_len = strlen(" (deleted)");
    const size_t len = file_spec.GetFilename().GetLength();
    file_spec.GetFilename().SetCStringWithLength(filename, len - deleted_len);
  }
  return error;
}

lldb::pid_t HostProcessPosix::GetProcessId() const { return m_process; }

bool HostProcessPosix::IsRunning() const {
  if (m_process == kInvalidPosixProcess)
    return false;

  // Send this process the null signal.  If it succeeds the process is running.
  Error error = Signal(0);
  return error.Success();
}

HostThread HostProcessPosix::StartMonitoring(
    const Host::MonitorChildProcessCallback &callback, bool monitor_signals) {
  return Host::StartMonitoringChildProcess(callback, m_process,
                                           monitor_signals);
}
