//===-- ProcessInfo.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
#include <climits>

// Other libraries and framework includes
// Project includes
#include "lldb/Target/ProcessInfo.h"

using namespace lldb;
using namespace lldb_private;

ProcessInfo::ProcessInfo()
    : m_executable(), m_arguments(), m_environment(), m_uid(UINT32_MAX),
      m_gid(UINT32_MAX), m_arch(), m_pid(LLDB_INVALID_PROCESS_ID) {}

ProcessInfo::ProcessInfo(const char *name, const ArchSpec &arch,
                         lldb::pid_t pid)
    : m_executable(name, false), m_arguments(), m_environment(),
      m_uid(UINT32_MAX), m_gid(UINT32_MAX), m_arch(arch), m_pid(pid) {}

void ProcessInfo::Clear() {
  m_executable.Clear();
  m_arguments.Clear();
  m_environment.Clear();
  m_uid = UINT32_MAX;
  m_gid = UINT32_MAX;
  m_arch.Clear();
  m_pid = LLDB_INVALID_PROCESS_ID;
}

const char *ProcessInfo::GetName() const {
  return m_executable.GetFilename().GetCString();
}

size_t ProcessInfo::GetNameLength() const {
  return m_executable.GetFilename().GetLength();
}

void ProcessInfo::SetExecutableFile(const FileSpec &exe_file,
                                    bool add_exe_file_as_first_arg) {
  if (exe_file) {
    m_executable = exe_file;
    if (add_exe_file_as_first_arg) {
      char filename[PATH_MAX];
      if (exe_file.GetPath(filename, sizeof(filename)))
        m_arguments.InsertArgumentAtIndex(0, filename);
    }
  } else {
    m_executable.Clear();
  }
}

const char *ProcessInfo::GetArg0() const {
  return (m_arg0.empty() ? nullptr : m_arg0.c_str());
}

void ProcessInfo::SetArg0(const char *arg) {
  if (arg && arg[0])
    m_arg0 = arg;
  else
    m_arg0.clear();
}

void ProcessInfo::SetArguments(char const **argv,
                               bool first_arg_is_executable) {
  m_arguments.SetArguments(argv);

  // Is the first argument the executable?
  if (first_arg_is_executable) {
    const char *first_arg = m_arguments.GetArgumentAtIndex(0);
    if (first_arg) {
      // Yes the first argument is an executable, set it as the executable
      // in the launch options. Don't resolve the file path as the path
      // could be a remote platform path
      const bool resolve = false;
      m_executable.SetFile(first_arg, resolve);
    }
  }
}

void ProcessInfo::SetArguments(const Args &args, bool first_arg_is_executable) {
  // Copy all arguments
  m_arguments = args;

  // Is the first argument the executable?
  if (first_arg_is_executable) {
    const char *first_arg = m_arguments.GetArgumentAtIndex(0);
    if (first_arg) {
      // Yes the first argument is an executable, set it as the executable
      // in the launch options. Don't resolve the file path as the path
      // could be a remote platform path
      const bool resolve = false;
      m_executable.SetFile(first_arg, resolve);
    }
  }
}
