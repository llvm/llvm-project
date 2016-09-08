//===-- Host.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C includes
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <sys/types.h>
#ifndef _WIN32
#include <dlfcn.h>
#include <grp.h>
#include <netdb.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#endif

#if defined(__linux__) || defined(__FreeBSD__) ||                              \
    defined(__FreeBSD_kernel__) || defined(__APPLE__) || defined(__NetBSD__)
#if !defined(__ANDROID__) && !defined(__ANDROID_NDK__)
#include <spawn.h>
#endif
#include <sys/syscall.h>
#include <sys/wait.h>
#endif

#if defined(__FreeBSD__)
#include <pthread_np.h>
#endif

// C++ Includes

// Other libraries and framework includes
// Project includes

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/MonitoringProcessLauncher.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/ProcessLauncher.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/CleanUp.h"
#include "lldb/lldb-private-forward.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"

#if defined(_WIN32)
#include "lldb/Host/windows/ProcessLauncherWindows.h"
#elif defined(__ANDROID__) || defined(__ANDROID_NDK__)
#include "lldb/Host/android/ProcessLauncherAndroid.h"
#else
#include "lldb/Host/posix/ProcessLauncherPosix.h"
#endif

#if defined(__APPLE__)
#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR 0x0100
#endif

extern "C" {
int __pthread_chdir(const char *path);
int __pthread_fchdir(int fildes);
}

#endif

using namespace lldb;
using namespace lldb_private;

#if !defined(__APPLE__) && !defined(_WIN32)
struct MonitorInfo {
  lldb::pid_t pid; // The process ID to monitor
  Host::MonitorChildProcessCallback
      callback; // The callback function to call when "pid" exits or signals
  void *callback_baton; // The callback baton for the callback function
  bool monitor_signals; // If true, call the callback when "pid" gets signaled.
};

static thread_result_t MonitorChildProcessThreadFunction(void *arg);

HostThread
Host::StartMonitoringChildProcess(Host::MonitorChildProcessCallback callback,
                                  void *callback_baton, lldb::pid_t pid,
                                  bool monitor_signals) {
  MonitorInfo *info_ptr = new MonitorInfo();

  info_ptr->pid = pid;
  info_ptr->callback = callback;
  info_ptr->callback_baton = callback_baton;
  info_ptr->monitor_signals = monitor_signals;

  char thread_name[256];
  ::snprintf(thread_name, sizeof(thread_name),
             "<lldb.host.wait4(pid=%" PRIu64 ")>", pid);
  return ThreadLauncher::LaunchThread(
      thread_name, MonitorChildProcessThreadFunction, info_ptr, NULL);
}

#ifndef __linux__
//------------------------------------------------------------------
// Scoped class that will disable thread canceling when it is
// constructed, and exception safely restore the previous value it
// when it goes out of scope.
//------------------------------------------------------------------
class ScopedPThreadCancelDisabler {
public:
  ScopedPThreadCancelDisabler() {
    // Disable the ability for this thread to be cancelled
    int err = ::pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, &m_old_state);
    if (err != 0)
      m_old_state = -1;
  }

  ~ScopedPThreadCancelDisabler() {
    // Restore the ability for this thread to be cancelled to what it
    // previously was.
    if (m_old_state != -1)
      ::pthread_setcancelstate(m_old_state, 0);
  }

private:
  int m_old_state; // Save the old cancelability state.
};
#endif // __linux__

#ifdef __linux__
#if defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
static __thread volatile sig_atomic_t g_usr1_called;
#else
static thread_local volatile sig_atomic_t g_usr1_called;
#endif

static void SigUsr1Handler(int) { g_usr1_called = 1; }
#endif // __linux__

static bool CheckForMonitorCancellation() {
#ifdef __linux__
  if (g_usr1_called) {
    g_usr1_called = 0;
    return true;
  }
#else
  ::pthread_testcancel();
#endif
  return false;
}

static thread_result_t MonitorChildProcessThreadFunction(void *arg) {
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
  const char *function = __FUNCTION__;
  if (log)
    log->Printf("%s (arg = %p) thread starting...", function, arg);

  MonitorInfo *info = (MonitorInfo *)arg;

  const Host::MonitorChildProcessCallback callback = info->callback;
  void *const callback_baton = info->callback_baton;
  const bool monitor_signals = info->monitor_signals;

  assert(info->pid <= UINT32_MAX);
  const ::pid_t pid = monitor_signals ? -1 * getpgid(info->pid) : info->pid;

  delete info;

  int status = -1;
#if defined(__FreeBSD__) || defined(__FreeBSD_kernel__)
#define __WALL 0
#endif
  const int options = __WALL;

#ifdef __linux__
  // This signal is only used to interrupt the thread from waitpid
  struct sigaction sigUsr1Action;
  memset(&sigUsr1Action, 0, sizeof(sigUsr1Action));
  sigUsr1Action.sa_handler = SigUsr1Handler;
  ::sigaction(SIGUSR1, &sigUsr1Action, nullptr);
#endif // __linux__

  while (1) {
    log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS);
    if (log)
      log->Printf("%s ::waitpid (pid = %" PRIi32 ", &status, options = %i)...",
                  function, pid, options);

    if (CheckForMonitorCancellation())
      break;

    // Get signals from all children with same process group of pid
    const ::pid_t wait_pid = ::waitpid(pid, &status, options);

    if (CheckForMonitorCancellation())
      break;

    if (wait_pid == -1) {
      if (errno == EINTR)
        continue;
      else {
        if (log)
          log->Printf(
              "%s (arg = %p) thread exiting because waitpid failed (%s)...",
              __FUNCTION__, arg, strerror(errno));
        break;
      }
    } else if (wait_pid > 0) {
      bool exited = false;
      int signal = 0;
      int exit_status = 0;
      const char *status_cstr = NULL;
      if (WIFSTOPPED(status)) {
        signal = WSTOPSIG(status);
        status_cstr = "STOPPED";
      } else if (WIFEXITED(status)) {
        exit_status = WEXITSTATUS(status);
        status_cstr = "EXITED";
        exited = true;
      } else if (WIFSIGNALED(status)) {
        signal = WTERMSIG(status);
        status_cstr = "SIGNALED";
        if (wait_pid == abs(pid)) {
          exited = true;
          exit_status = -1;
        }
      } else {
        status_cstr = "(\?\?\?)";
      }

      // Scope for pthread_cancel_disabler
      {
#ifndef __linux__
        ScopedPThreadCancelDisabler pthread_cancel_disabler;
#endif

        log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS);
        if (log)
          log->Printf("%s ::waitpid (pid = %" PRIi32
                      ", &status, options = %i) => pid = %" PRIi32
                      ", status = 0x%8.8x (%s), signal = %i, exit_state = %i",
                      function, pid, options, wait_pid, status, status_cstr,
                      signal, exit_status);

        if (exited || (signal != 0 && monitor_signals)) {
          bool callback_return = false;
          if (callback)
            callback_return =
                callback(callback_baton, wait_pid, exited, signal, exit_status);

          // If our process exited, then this thread should exit
          if (exited && wait_pid == abs(pid)) {
            if (log)
              log->Printf("%s (arg = %p) thread exiting because pid received "
                          "exit signal...",
                          __FUNCTION__, arg);
            break;
          }
          // If the callback returns true, it means this process should
          // exit
          if (callback_return) {
            if (log)
              log->Printf("%s (arg = %p) thread exiting because callback "
                          "returned true...",
                          __FUNCTION__, arg);
            break;
          }
        }
      }
    }
  }

  log = lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS);
  if (log)
    log->Printf("%s (arg = %p) thread exiting...", __FUNCTION__, arg);

  return NULL;
}

#endif // #if !defined (__APPLE__) && !defined (_WIN32)

#if !defined(__APPLE__)

void Host::SystemLog(SystemLogType type, const char *format, va_list args) {
  vfprintf(stderr, format, args);
}

#endif

void Host::SystemLog(SystemLogType type, const char *format, ...) {
  va_list args;
  va_start(args, format);
  SystemLog(type, format, args);
  va_end(args);
}

lldb::pid_t Host::GetCurrentProcessID() { return ::getpid(); }

#ifndef _WIN32

lldb::tid_t Host::GetCurrentThreadID() {
#if defined(__APPLE__)
  // Calling "mach_thread_self()" bumps the reference count on the thread
  // port, so we need to deallocate it. mach_task_self() doesn't bump the ref
  // count.
  thread_port_t thread_self = mach_thread_self();
  mach_port_deallocate(mach_task_self(), thread_self);
  return thread_self;
#elif defined(__FreeBSD__)
  return lldb::tid_t(pthread_getthreadid_np());
#elif defined(__ANDROID_NDK__)
  return lldb::tid_t(gettid());
#elif defined(__linux__)
  return lldb::tid_t(syscall(SYS_gettid));
#else
  return lldb::tid_t(pthread_self());
#endif
}

lldb::thread_t Host::GetCurrentThread() {
  return lldb::thread_t(pthread_self());
}

const char *Host::GetSignalAsCString(int signo) {
  switch (signo) {
  case SIGHUP:
    return "SIGHUP"; // 1    hangup
  case SIGINT:
    return "SIGINT"; // 2    interrupt
  case SIGQUIT:
    return "SIGQUIT"; // 3    quit
  case SIGILL:
    return "SIGILL"; // 4    illegal instruction (not reset when caught)
  case SIGTRAP:
    return "SIGTRAP"; // 5    trace trap (not reset when caught)
  case SIGABRT:
    return "SIGABRT"; // 6    abort()
#if defined(SIGPOLL)
#if !defined(SIGIO) || (SIGPOLL != SIGIO)
  // Under some GNU/Linux, SIGPOLL and SIGIO are the same. Causing the build to
  // fail with 'multiple define cases with same value'
  case SIGPOLL:
    return "SIGPOLL"; // 7    pollable event ([XSR] generated, not supported)
#endif
#endif
#if defined(SIGEMT)
  case SIGEMT:
    return "SIGEMT"; // 7    EMT instruction
#endif
  case SIGFPE:
    return "SIGFPE"; // 8    floating point exception
  case SIGKILL:
    return "SIGKILL"; // 9    kill (cannot be caught or ignored)
  case SIGBUS:
    return "SIGBUS"; // 10    bus error
  case SIGSEGV:
    return "SIGSEGV"; // 11    segmentation violation
  case SIGSYS:
    return "SIGSYS"; // 12    bad argument to system call
  case SIGPIPE:
    return "SIGPIPE"; // 13    write on a pipe with no one to read it
  case SIGALRM:
    return "SIGALRM"; // 14    alarm clock
  case SIGTERM:
    return "SIGTERM"; // 15    software termination signal from kill
  case SIGURG:
    return "SIGURG"; // 16    urgent condition on IO channel
  case SIGSTOP:
    return "SIGSTOP"; // 17    sendable stop signal not from tty
  case SIGTSTP:
    return "SIGTSTP"; // 18    stop signal from tty
  case SIGCONT:
    return "SIGCONT"; // 19    continue a stopped process
  case SIGCHLD:
    return "SIGCHLD"; // 20    to parent on child stop or exit
  case SIGTTIN:
    return "SIGTTIN"; // 21    to readers pgrp upon background tty read
  case SIGTTOU:
    return "SIGTTOU"; // 22    like TTIN for output if (tp->t_local&LTOSTOP)
#if defined(SIGIO)
  case SIGIO:
    return "SIGIO"; // 23    input/output possible signal
#endif
  case SIGXCPU:
    return "SIGXCPU"; // 24    exceeded CPU time limit
  case SIGXFSZ:
    return "SIGXFSZ"; // 25    exceeded file size limit
  case SIGVTALRM:
    return "SIGVTALRM"; // 26    virtual time alarm
  case SIGPROF:
    return "SIGPROF"; // 27    profiling time alarm
#if defined(SIGWINCH)
  case SIGWINCH:
    return "SIGWINCH"; // 28    window size changes
#endif
#if defined(SIGINFO)
  case SIGINFO:
    return "SIGINFO"; // 29    information request
#endif
  case SIGUSR1:
    return "SIGUSR1"; // 30    user defined signal 1
  case SIGUSR2:
    return "SIGUSR2"; // 31    user defined signal 2
  default:
    break;
  }
  return NULL;
}

#endif

#ifndef _WIN32

lldb::thread_key_t
Host::ThreadLocalStorageCreate(ThreadLocalStorageCleanupCallback callback) {
  pthread_key_t key;
  ::pthread_key_create(&key, callback);
  return key;
}

void *Host::ThreadLocalStorageGet(lldb::thread_key_t key) {
  return ::pthread_getspecific(key);
}

void Host::ThreadLocalStorageSet(lldb::thread_key_t key, void *value) {
  ::pthread_setspecific(key, value);
}

#endif

#if !defined(__APPLE__) // see Host.mm

bool Host::GetBundleDirectory(const FileSpec &file, FileSpec &bundle) {
  bundle.Clear();
  return false;
}

bool Host::ResolveExecutableInBundle(FileSpec &file) { return false; }
#endif

#ifndef _WIN32

FileSpec Host::GetModuleFileSpecForHostAddress(const void *host_addr) {
  FileSpec module_filespec;
#if !defined(__ANDROID__) && !defined(__ANDROID_NDK__)
  Dl_info info;
  if (::dladdr(host_addr, &info)) {
    if (info.dli_fname)
      module_filespec.SetFile(info.dli_fname, true);
  }
#endif
  return module_filespec;
}

#endif

#if !defined(__linux__)
bool Host::FindProcessThreads(const lldb::pid_t pid, TidMap &tids_to_attach) {
  return false;
}
#endif

struct ShellInfo {
  ShellInfo()
      : process_reaped(false), can_delete(false), pid(LLDB_INVALID_PROCESS_ID),
        signo(-1), status(-1) {}

  lldb_private::Predicate<bool> process_reaped;
  lldb_private::Predicate<bool> can_delete;
  lldb::pid_t pid;
  int signo;
  int status;
};

static bool
MonitorShellCommand(void *callback_baton, lldb::pid_t pid,
                    bool exited, // True if the process did exit
                    int signo,   // Zero for no signal
                    int status)  // Exit value of process if signal is zero
{
  ShellInfo *shell_info = (ShellInfo *)callback_baton;
  shell_info->pid = pid;
  shell_info->signo = signo;
  shell_info->status = status;
  // Let the thread running Host::RunShellCommand() know that the process
  // exited and that ShellInfo has been filled in by broadcasting to it
  shell_info->process_reaped.SetValue(1, eBroadcastAlways);
  // Now wait for a handshake back from that thread running
  // Host::RunShellCommand
  // so we know that we can delete shell_info_ptr
  shell_info->can_delete.WaitForValueEqualTo(true);
  // Sleep a bit to allow the shell_info->can_delete.SetValue() to complete...
  usleep(1000);
  // Now delete the shell info that was passed into this function
  delete shell_info;
  return true;
}

Error Host::RunShellCommand(const char *command, const FileSpec &working_dir,
                            int *status_ptr, int *signo_ptr,
                            std::string *command_output_ptr,
                            uint32_t timeout_sec, bool run_in_default_shell) {
  return RunShellCommand(Args(command), working_dir, status_ptr, signo_ptr,
                         command_output_ptr, timeout_sec, run_in_default_shell);
}

Error Host::RunShellCommand(const Args &args, const FileSpec &working_dir,
                            int *status_ptr, int *signo_ptr,
                            std::string *command_output_ptr,
                            uint32_t timeout_sec, bool run_in_default_shell) {
  Error error;
  ProcessLaunchInfo launch_info;
  launch_info.SetArchitecture(HostInfo::GetArchitecture());
  if (run_in_default_shell) {
    // Run the command in a shell
    launch_info.SetShell(HostInfo::GetDefaultShell());
    launch_info.GetArguments().AppendArguments(args);
    const bool localhost = true;
    const bool will_debug = false;
    const bool first_arg_is_full_shell_command = false;
    launch_info.ConvertArgumentsForLaunchingInShell(
        error, localhost, will_debug, first_arg_is_full_shell_command, 0);
  } else {
    // No shell, just run it
    const bool first_arg_is_executable = true;
    launch_info.SetArguments(args, first_arg_is_executable);
  }

  if (working_dir)
    launch_info.SetWorkingDirectory(working_dir);
  llvm::SmallString<PATH_MAX> output_file_path;

  if (command_output_ptr) {
    // Create a temporary file to get the stdout/stderr and redirect the
    // output of the command into this file. We will later read this file
    // if all goes well and fill the data into "command_output_ptr"
    FileSpec tmpdir_file_spec;
    if (HostInfo::GetLLDBPath(ePathTypeLLDBTempSystemDir, tmpdir_file_spec)) {
      tmpdir_file_spec.AppendPathComponent("lldb-shell-output.%%%%%%");
      llvm::sys::fs::createUniqueFile(tmpdir_file_spec.GetPath().c_str(),
                                      output_file_path);
    } else {
      llvm::sys::fs::createTemporaryFile("lldb-shell-output.%%%%%%", "",
                                         output_file_path);
    }
  }

  FileSpec output_file_spec{output_file_path.c_str(), false};

  launch_info.AppendSuppressFileAction(STDIN_FILENO, true, false);
  if (output_file_spec) {
    launch_info.AppendOpenFileAction(STDOUT_FILENO, output_file_spec, false,
                                     true);
    launch_info.AppendDuplicateFileAction(STDOUT_FILENO, STDERR_FILENO);
  } else {
    launch_info.AppendSuppressFileAction(STDOUT_FILENO, false, true);
    launch_info.AppendSuppressFileAction(STDERR_FILENO, false, true);
  }

  // The process monitor callback will delete the 'shell_info_ptr' below...
  std::unique_ptr<ShellInfo> shell_info_ap(new ShellInfo());

  const bool monitor_signals = false;
  launch_info.SetMonitorProcessCallback(MonitorShellCommand,
                                        shell_info_ap.get(), monitor_signals);

  error = LaunchProcess(launch_info);
  const lldb::pid_t pid = launch_info.GetProcessID();

  if (error.Success() && pid == LLDB_INVALID_PROCESS_ID)
    error.SetErrorString("failed to get process ID");

  if (error.Success()) {
    // The process successfully launched, so we can defer ownership of
    // "shell_info" to the MonitorShellCommand callback function that will
    // get called when the process dies. We release the unique pointer as it
    // doesn't need to delete the ShellInfo anymore.
    ShellInfo *shell_info = shell_info_ap.release();
    TimeValue *timeout_ptr = nullptr;
    TimeValue timeout_time(TimeValue::Now());
    if (timeout_sec > 0) {
      timeout_time.OffsetWithSeconds(timeout_sec);
      timeout_ptr = &timeout_time;
    }
    bool timed_out = false;
    shell_info->process_reaped.WaitForValueEqualTo(true, timeout_ptr,
                                                   &timed_out);
    if (timed_out) {
      error.SetErrorString("timed out waiting for shell command to complete");

      // Kill the process since it didn't complete within the timeout specified
      Kill(pid, SIGKILL);
      // Wait for the monitor callback to get the message
      timeout_time = TimeValue::Now();
      timeout_time.OffsetWithSeconds(1);
      timed_out = false;
      shell_info->process_reaped.WaitForValueEqualTo(true, &timeout_time,
                                                     &timed_out);
    } else {
      if (status_ptr)
        *status_ptr = shell_info->status;

      if (signo_ptr)
        *signo_ptr = shell_info->signo;

      if (command_output_ptr) {
        command_output_ptr->clear();
        uint64_t file_size = output_file_spec.GetByteSize();
        if (file_size > 0) {
          if (file_size > command_output_ptr->max_size()) {
            error.SetErrorStringWithFormat(
                "shell command output is too large to fit into a std::string");
          } else {
            std::vector<char> command_output(file_size);
            output_file_spec.ReadFileContents(0, command_output.data(),
                                              file_size, &error);
            if (error.Success())
              command_output_ptr->assign(command_output.data(), file_size);
          }
        }
      }
    }
    shell_info->can_delete.SetValue(true, eBroadcastAlways);
  }

  if (FileSystem::GetFileExists(output_file_spec))
    FileSystem::Unlink(output_file_spec);
  // Handshake with the monitor thread, or just let it know in advance that
  // it can delete "shell_info" in case we timed out and were not able to kill
  // the process...
  return error;
}

// LaunchProcessPosixSpawn for Apple, Linux, FreeBSD and other GLIBC
// systems

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__) ||        \
    defined(__GLIBC__) || defined(__NetBSD__)
#if !defined(__ANDROID__) && !defined(__ANDROID_NDK__)
// this method needs to be visible to macosx/Host.cpp and
// common/Host.cpp.

short Host::GetPosixspawnFlags(const ProcessLaunchInfo &launch_info) {
  short flags = POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;

#if defined(__APPLE__)
  if (launch_info.GetFlags().Test(eLaunchFlagExec))
    flags |= POSIX_SPAWN_SETEXEC; // Darwin specific posix_spawn flag

  if (launch_info.GetFlags().Test(eLaunchFlagDebug))
    flags |= POSIX_SPAWN_START_SUSPENDED; // Darwin specific posix_spawn flag

  if (launch_info.GetFlags().Test(eLaunchFlagDisableASLR))
    flags |= _POSIX_SPAWN_DISABLE_ASLR; // Darwin specific posix_spawn flag

  if (launch_info.GetLaunchInSeparateProcessGroup())
    flags |= POSIX_SPAWN_SETPGROUP;

#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
#if defined(__APPLE__) && (defined(__x86_64__) || defined(__i386__))
  static LazyBool g_use_close_on_exec_flag = eLazyBoolCalculate;
  if (g_use_close_on_exec_flag == eLazyBoolCalculate) {
    g_use_close_on_exec_flag = eLazyBoolNo;

    uint32_t major, minor, update;
    if (HostInfo::GetOSVersion(major, minor, update)) {
      // Kernel panic if we use the POSIX_SPAWN_CLOEXEC_DEFAULT on 10.7 or
      // earlier
      if (major > 10 || (major == 10 && minor > 7)) {
        // Only enable for 10.8 and later OS versions
        g_use_close_on_exec_flag = eLazyBoolYes;
      }
    }
  }
#else
  static LazyBool g_use_close_on_exec_flag = eLazyBoolYes;
#endif
  // Close all files exception those with file actions if this is supported.
  if (g_use_close_on_exec_flag == eLazyBoolYes)
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
#endif // #if defined (__APPLE__)
  return flags;
}

Error Host::LaunchProcessPosixSpawn(const char *exe_path,
                                    const ProcessLaunchInfo &launch_info,
                                    lldb::pid_t &pid) {
  Error error;
  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_HOST |
                                                  LIBLLDB_LOG_PROCESS));

  posix_spawnattr_t attr;
  error.SetError(::posix_spawnattr_init(&attr), eErrorTypePOSIX);

  if (error.Fail() || log)
    error.PutToLog(log, "::posix_spawnattr_init ( &attr )");
  if (error.Fail())
    return error;

  // Make a quick class that will cleanup the posix spawn attributes in case
  // we return in the middle of this function.
  lldb_utility::CleanUp<posix_spawnattr_t *, int> posix_spawnattr_cleanup(
      &attr, posix_spawnattr_destroy);

  sigset_t no_signals;
  sigset_t all_signals;
  sigemptyset(&no_signals);
  sigfillset(&all_signals);
  ::posix_spawnattr_setsigmask(&attr, &no_signals);
#if defined(__linux__) || defined(__FreeBSD__)
  ::posix_spawnattr_setsigdefault(&attr, &no_signals);
#else
  ::posix_spawnattr_setsigdefault(&attr, &all_signals);
#endif

  short flags = GetPosixspawnFlags(launch_info);

  error.SetError(::posix_spawnattr_setflags(&attr, flags), eErrorTypePOSIX);
  if (error.Fail() || log)
    error.PutToLog(log, "::posix_spawnattr_setflags ( &attr, flags=0x%8.8x )",
                   flags);
  if (error.Fail())
    return error;

// posix_spawnattr_setbinpref_np appears to be an Apple extension per:
// http://www.unix.com/man-page/OSX/3/posix_spawnattr_setbinpref_np/
#if defined(__APPLE__) && !defined(__arm__)

  // Don't set the binpref if a shell was provided.  After all, that's only
  // going to affect what version of the shell
  // is launched, not what fork of the binary is launched.  We insert "arch
  // --arch <ARCH> as part of the shell invocation
  // to do that job on OSX.

  if (launch_info.GetShell() == nullptr) {
    // We don't need to do this for ARM, and we really shouldn't now that we
    // have multiple CPU subtypes and no posix_spawnattr call that allows us
    // to set which CPU subtype to launch...
    const ArchSpec &arch_spec = launch_info.GetArchitecture();
    cpu_type_t cpu = arch_spec.GetMachOCPUType();
    cpu_type_t sub = arch_spec.GetMachOCPUSubType();
    if (cpu != 0 && cpu != static_cast<cpu_type_t>(UINT32_MAX) &&
        cpu != static_cast<cpu_type_t>(LLDB_INVALID_CPUTYPE) &&
        !(cpu == 0x01000007 && sub == 8)) // If haswell is specified, don't try
                                          // to set the CPU type or we will fail
    {
      size_t ocount = 0;
      error.SetError(::posix_spawnattr_setbinpref_np(&attr, 1, &cpu, &ocount),
                     eErrorTypePOSIX);
      if (error.Fail() || log)
        error.PutToLog(log, "::posix_spawnattr_setbinpref_np ( &attr, 1, "
                            "cpu_type = 0x%8.8x, count => %llu )",
                       cpu, (uint64_t)ocount);

      if (error.Fail() || ocount != 1)
        return error;
    }
  }

#endif

  const char *tmp_argv[2];
  char *const *argv = const_cast<char *const *>(
      launch_info.GetArguments().GetConstArgumentVector());
  char *const *envp = const_cast<char *const *>(
      launch_info.GetEnvironmentEntries().GetConstArgumentVector());
  if (argv == NULL) {
    // posix_spawn gets very unhappy if it doesn't have at least the program
    // name in argv[0]. One of the side affects I have noticed is the
    // environment
    // variables don't make it into the child process if "argv == NULL"!!!
    tmp_argv[0] = exe_path;
    tmp_argv[1] = NULL;
    argv = const_cast<char *const *>(tmp_argv);
  }

#if !defined(__APPLE__)
  // manage the working directory
  char current_dir[PATH_MAX];
  current_dir[0] = '\0';
#endif

  FileSpec working_dir{launch_info.GetWorkingDirectory()};
  if (working_dir) {
#if defined(__APPLE__)
    // Set the working directory on this thread only
    if (__pthread_chdir(working_dir.GetCString()) < 0) {
      if (errno == ENOENT) {
        error.SetErrorStringWithFormat("No such file or directory: %s",
                                       working_dir.GetCString());
      } else if (errno == ENOTDIR) {
        error.SetErrorStringWithFormat("Path doesn't name a directory: %s",
                                       working_dir.GetCString());
      } else {
        error.SetErrorStringWithFormat("An unknown error occurred when "
                                       "changing directory for process "
                                       "execution.");
      }
      return error;
    }
#else
    if (::getcwd(current_dir, sizeof(current_dir)) == NULL) {
      error.SetError(errno, eErrorTypePOSIX);
      error.LogIfError(log, "unable to save the current directory");
      return error;
    }

    if (::chdir(working_dir.GetCString()) == -1) {
      error.SetError(errno, eErrorTypePOSIX);
      error.LogIfError(log, "unable to change working directory to %s",
                       working_dir.GetCString());
      return error;
    }
#endif
  }

  ::pid_t result_pid = LLDB_INVALID_PROCESS_ID;
  const size_t num_file_actions = launch_info.GetNumFileActions();
  if (num_file_actions > 0) {
    posix_spawn_file_actions_t file_actions;
    error.SetError(::posix_spawn_file_actions_init(&file_actions),
                   eErrorTypePOSIX);
    if (error.Fail() || log)
      error.PutToLog(log, "::posix_spawn_file_actions_init ( &file_actions )");
    if (error.Fail())
      return error;

    // Make a quick class that will cleanup the posix spawn attributes in case
    // we return in the middle of this function.
    lldb_utility::CleanUp<posix_spawn_file_actions_t *, int>
        posix_spawn_file_actions_cleanup(&file_actions,
                                         posix_spawn_file_actions_destroy);

    for (size_t i = 0; i < num_file_actions; ++i) {
      const FileAction *launch_file_action =
          launch_info.GetFileActionAtIndex(i);
      if (launch_file_action) {
        if (!AddPosixSpawnFileAction(&file_actions, launch_file_action, log,
                                     error))
          return error;
      }
    }

    error.SetError(
        ::posix_spawnp(&result_pid, exe_path, &file_actions, &attr, argv, envp),
        eErrorTypePOSIX);

    if (error.Fail() || log) {
      error.PutToLog(
          log, "::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, "
               "attr = %p, argv = %p, envp = %p )",
          result_pid, exe_path, static_cast<void *>(&file_actions),
          static_cast<void *>(&attr), reinterpret_cast<const void *>(argv),
          reinterpret_cast<const void *>(envp));
      if (log) {
        for (int ii = 0; argv[ii]; ++ii)
          log->Printf("argv[%i] = '%s'", ii, argv[ii]);
      }
    }

  } else {
    error.SetError(
        ::posix_spawnp(&result_pid, exe_path, NULL, &attr, argv, envp),
        eErrorTypePOSIX);

    if (error.Fail() || log) {
      error.PutToLog(log, "::posix_spawnp ( pid => %i, path = '%s', "
                          "file_actions = NULL, attr = %p, argv = %p, envp = "
                          "%p )",
                     result_pid, exe_path, static_cast<void *>(&attr),
                     reinterpret_cast<const void *>(argv),
                     reinterpret_cast<const void *>(envp));
      if (log) {
        for (int ii = 0; argv[ii]; ++ii)
          log->Printf("argv[%i] = '%s'", ii, argv[ii]);
      }
    }
  }
  pid = result_pid;

  if (working_dir) {
#if defined(__APPLE__)
    // No more thread specific current working directory
    __pthread_fchdir(-1);
#else
    if (::chdir(current_dir) == -1 && error.Success()) {
      error.SetError(errno, eErrorTypePOSIX);
      error.LogIfError(log, "unable to change current directory back to %s",
                       current_dir);
    }
#endif
  }

  return error;
}

bool Host::AddPosixSpawnFileAction(void *_file_actions, const FileAction *info,
                                   Log *log, Error &error) {
  if (info == NULL)
    return false;

  posix_spawn_file_actions_t *file_actions =
      reinterpret_cast<posix_spawn_file_actions_t *>(_file_actions);

  switch (info->GetAction()) {
  case FileAction::eFileActionNone:
    error.Clear();
    break;

  case FileAction::eFileActionClose:
    if (info->GetFD() == -1)
      error.SetErrorString(
          "invalid fd for posix_spawn_file_actions_addclose(...)");
    else {
      error.SetError(
          ::posix_spawn_file_actions_addclose(file_actions, info->GetFD()),
          eErrorTypePOSIX);
      if (log && (error.Fail() || log))
        error.PutToLog(log,
                       "posix_spawn_file_actions_addclose (action=%p, fd=%i)",
                       static_cast<void *>(file_actions), info->GetFD());
    }
    break;

  case FileAction::eFileActionDuplicate:
    if (info->GetFD() == -1)
      error.SetErrorString(
          "invalid fd for posix_spawn_file_actions_adddup2(...)");
    else if (info->GetActionArgument() == -1)
      error.SetErrorString(
          "invalid duplicate fd for posix_spawn_file_actions_adddup2(...)");
    else {
      error.SetError(
          ::posix_spawn_file_actions_adddup2(file_actions, info->GetFD(),
                                             info->GetActionArgument()),
          eErrorTypePOSIX);
      if (log && (error.Fail() || log))
        error.PutToLog(
            log,
            "posix_spawn_file_actions_adddup2 (action=%p, fd=%i, dup_fd=%i)",
            static_cast<void *>(file_actions), info->GetFD(),
            info->GetActionArgument());
    }
    break;

  case FileAction::eFileActionOpen:
    if (info->GetFD() == -1)
      error.SetErrorString(
          "invalid fd in posix_spawn_file_actions_addopen(...)");
    else {
      int oflag = info->GetActionArgument();

      mode_t mode = 0;

      if (oflag & O_CREAT)
        mode = 0640;

      error.SetError(
          ::posix_spawn_file_actions_addopen(file_actions, info->GetFD(),
                                             info->GetPath(), oflag, mode),
          eErrorTypePOSIX);
      if (error.Fail() || log)
        error.PutToLog(log, "posix_spawn_file_actions_addopen (action=%p, "
                            "fd=%i, path='%s', oflag=%i, mode=%i)",
                       static_cast<void *>(file_actions), info->GetFD(),
                       info->GetPath(), oflag, mode);
    }
    break;
  }
  return error.Success();
}
#endif // !defined(__ANDROID__) && !defined(__ANDROID_NDK__)
#endif // defined (__APPLE__) || defined (__linux__) || defined (__FreeBSD__) ||
       // defined (__GLIBC__) || defined(__NetBSD__)

#if defined(__linux__) || defined(__FreeBSD__) || defined(__GLIBC__) ||        \
    defined(__NetBSD__) || defined(_WIN32)
// The functions below implement process launching via posix_spawn() for Linux,
// FreeBSD and NetBSD.

Error Host::LaunchProcess(ProcessLaunchInfo &launch_info) {
  std::unique_ptr<ProcessLauncher> delegate_launcher;
#if defined(_WIN32)
  delegate_launcher.reset(new ProcessLauncherWindows());
#elif defined(__ANDROID__) || defined(__ANDROID_NDK__)
  delegate_launcher.reset(new ProcessLauncherAndroid());
#else
  delegate_launcher.reset(new ProcessLauncherPosix());
#endif
  MonitoringProcessLauncher launcher(std::move(delegate_launcher));

  Error error;
  HostProcess process = launcher.LaunchProcess(launch_info, error);

  // TODO(zturner): It would be better if the entire HostProcess were returned
  // instead of writing
  // it into this structure.
  launch_info.SetProcessID(process.GetProcessId());

  return error;
}
#endif // defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__)

#ifndef _WIN32
void Host::Kill(lldb::pid_t pid, int signo) { ::kill(pid, signo); }

#endif

#if !defined(__APPLE__)
bool Host::OpenFileInExternalEditor(const FileSpec &file_spec,
                                    uint32_t line_no) {
  return false;
}

void Host::SetCrashDescriptionWithFormat(const char *format, ...) {}

void Host::SetCrashDescription(const char *description) {}

#endif

const UnixSignalsSP &Host::GetUnixSignals() {
  static const auto s_unix_signals_sp =
      UnixSignals::Create(HostInfo::GetArchitecture());
  return s_unix_signals_sp;
}
