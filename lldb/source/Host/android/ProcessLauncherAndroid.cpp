//===-- ProcessLauncherAndroid.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/android/ProcessLauncherAndroid.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"

#include "lldb/Target/ProcessLaunchInfo.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

static bool DupDescriptor(const FileSpec &file_spec, int fd, int flags) {
  int target_fd = ::open(file_spec.GetCString(), flags, 0666);

  if (target_fd == -1)
    return false;

  if (::dup2(target_fd, fd) == -1)
    return false;

  return (::close(target_fd) == -1) ? false : true;
}

// If there is no PATH variable specified inside the environment then set the
// path to /system/bin.
// It is required because the default path used by execve() is wrong on android.
static void FixupEnvironment(Args &env) {
  static const char *path = "PATH=";
  static const int path_len = ::strlen(path);
  for (const char **args = env.GetConstArgumentVector(); *args; ++args)
    if (::strncmp(path, *args, path_len) == 0)
      return;
  env.AppendArgument("PATH=/system/bin");
}

HostProcess
ProcessLauncherAndroid::LaunchProcess(const ProcessLaunchInfo &launch_info,
                                      Error &error) {
  // TODO: Handle other launch parameters specified in launc_info

  char exe_path[PATH_MAX];
  launch_info.GetExecutableFile().GetPath(exe_path, sizeof(exe_path));

  lldb::pid_t pid = ::fork();
  if (pid == static_cast<lldb::pid_t>(-1)) {
    // Fork failed
    error.SetErrorStringWithFormat("Fork failed with error message: %s",
                                   strerror(errno));
    return HostProcess(LLDB_INVALID_PROCESS_ID);
  } else if (pid == 0) {
    if (const lldb_private::FileAction *file_action =
            launch_info.GetFileActionForFD(STDIN_FILENO)) {
      FileSpec file_spec = file_action->GetFileSpec();
      if (file_spec)
        if (!DupDescriptor(file_spec, STDIN_FILENO, O_RDONLY))
          exit(-1);
    }

    if (const lldb_private::FileAction *file_action =
            launch_info.GetFileActionForFD(STDOUT_FILENO)) {
      FileSpec file_spec = file_action->GetFileSpec();
      if (file_spec)
        if (!DupDescriptor(file_spec, STDOUT_FILENO,
                           O_WRONLY | O_CREAT | O_TRUNC))
          exit(-1);
    }

    if (const lldb_private::FileAction *file_action =
            launch_info.GetFileActionForFD(STDERR_FILENO)) {
      FileSpec file_spec = file_action->GetFileSpec();
      if (file_spec)
        if (!DupDescriptor(file_spec, STDERR_FILENO,
                           O_WRONLY | O_CREAT | O_TRUNC))
          exit(-1);
    }

    // Child process
    const char **argv = launch_info.GetArguments().GetConstArgumentVector();

    Args env = launch_info.GetEnvironmentEntries();
    FixupEnvironment(env);
    const char **envp = env.GetConstArgumentVector();

    FileSpec working_dir = launch_info.GetWorkingDirectory();
    if (working_dir) {
      if (::chdir(working_dir.GetCString()) != 0)
        exit(-1);
    }

    execve(argv[0], const_cast<char *const *>(argv),
           const_cast<char *const *>(envp));
    exit(-1);
  }

  return HostProcess(pid);
}
