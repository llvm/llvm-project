//===-- source/Host/qnx/Host.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <dirent.h>

#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/qnx/Support.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/ProcessInfo.h"

#include "llvm/Object/ELF.h"

extern "C" {
extern char **environ;
}

using namespace lldb;
using namespace lldb_private;

static bool GetExecutableFile(::pid_t pid, ProcessInstanceInfo &process_info) {
  auto buffer_or_error = getProcFile(pid, "exefile");

  if (!buffer_or_error) {
    return false;
  }

  llvm::StringRef exe_path = buffer_or_error.get()->getBuffer();

  if (exe_path.empty()) {
    return false;
  }

  process_info.GetExecutableFile().SetFile(exe_path, FileSpec::Style::native);

  return true;
}

static bool GetArchitecture(ProcessInstanceInfo &process_info) {
  Log *log = GetLog(LLDBLog::Host);

  FileSpec executable_file = process_info.GetExecutableFile();

  auto buffer_sp =
      FileSystem::Instance().CreateDataBuffer(executable_file, 0x20, 0);
  if (!buffer_sp) {
    process_info.SetArchitecture(ArchSpec());
    return false;
  }

  uint8_t exe_class =
      llvm::object::getElfArchType(
          {reinterpret_cast<const char *>(buffer_sp->GetBytes()),
           size_t(buffer_sp->GetByteSize())})
          .first;

  switch (exe_class) {
  case llvm::ELF::ELFCLASS32:
    process_info.SetArchitecture(
        HostInfo::GetArchitecture(HostInfo::eArchKind32));
    return true;
  case llvm::ELF::ELFCLASS64:
    process_info.SetArchitecture(
        HostInfo::GetArchitecture(HostInfo::eArchKind64));
    return true;
  default:
    LLDB_LOG(log, "Unknown elf class ({0}) in file {1}", exe_class,
             executable_file.GetPath());
    process_info.SetArchitecture(ArchSpec());
    return false;
  }
}

static bool GetProcessArgs(::pid_t pid, ProcessInstanceInfo &process_info) {
  auto buffer_or_error = getProcFile(pid, "cmdline");

  if (!buffer_or_error)
    return false;

  std::unique_ptr<llvm::MemoryBuffer> cmdline = std::move(*buffer_or_error);

  llvm::StringRef arg0, rest;
  std::tie(arg0, rest) = cmdline->getBuffer().split('\0');
  process_info.SetArg0(arg0);
  while (!rest.empty()) {
    llvm::StringRef arg;
    std::tie(arg, rest) = rest.split('\0');
    process_info.GetArguments().AppendArgument(arg);
  }

  return true;
}

static bool IsDirNumeric(const char *dname) {
  for (; *dname; dname++) {
    if (!isdigit(*dname))
      return false;
  }
  return true;
}

uint32_t Host::FindProcessesImpl(const ProcessInstanceInfoMatch &match_info,
                                 ProcessInstanceInfoList &process_infos) {
  static const std::string procdir = "/proc/";

  DIR *dirproc = opendir(procdir.c_str());
  if (dirproc) {
    struct dirent *direntry = nullptr;
    const lldb::pid_t our_pid = getpid();

    while ((direntry = readdir(dirproc)) != nullptr) {
      struct stat statp;
      std::string path = procdir + std::string(direntry->d_name);

      if (stat(path.c_str(), &statp) == -1)
        continue;

      if (!S_ISDIR(statp.st_mode) || !IsDirNumeric(direntry->d_name))
        continue;

      lldb::pid_t pid = std::stoi(direntry->d_name);

      // Skip this process.
      if (pid == our_pid)
        continue;

      ProcessInstanceInfo process_info;
      if (!GetProcessInfo(pid, process_info))
        continue;

      // TODO: Skip if process is under debug.

      // TODO: Skip if process is a zombie.

      // TODO: Match user if we're not matching all users and not running as
      // root.

      if (match_info.Matches(process_info)) {
        process_infos.push_back(process_info);
      }
    }

    closedir(dirproc);
  }

  return process_infos.size();
}

bool Host::GetProcessInfo(lldb::pid_t pid, ProcessInstanceInfo &process_info) {
  Log *log = GetLog(LLDBLog::Host);

  process_info.SetProcessID(pid);

  int ret = true;

  if (!GetExecutableFile(pid, process_info)) {
    LLDB_LOG(log, "Failed to retrieve {0}'s executable file", pid);
    ret = false;
  }

  if (!GetArchitecture(process_info)) {
    LLDB_LOG(log, "Failed to retrieve {0}'s architecture", pid);
    ret = false;
  }

  if (!GetProcessArgs(pid, process_info)) {
    LLDB_LOG(log, "Failed to retrieve {0}'s arguments", pid);
    ret = false;
  }

  // TODO: Get the process's environment, state, real group ID, effective group
  // ID, user ID, effective user ID, and parent's process ID.

  return ret;
}

Environment Host::GetEnvironment() { return Environment(environ); }
