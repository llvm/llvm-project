//===-- SingleStepCheck.cpp ----------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SingleStepCheck.h"

#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include "NativeProcessLinux.h"

#include "llvm/Support/Compiler.h"

#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/linux/Ptrace.h"

using namespace lldb_private::process_linux;

#if defined(__arm64__) || defined(__aarch64__)
namespace {

void LLVM_ATTRIBUTE_NORETURN Child() {
  if (ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) == -1)
    _exit(1);

  // We just do an endless loop SIGSTOPPING ourselves until killed. The tracer
  // will fiddle with our cpu
  // affinities and monitor the behaviour.
  for (;;) {
    raise(SIGSTOP);

    // Generate a bunch of instructions here, so that a single-step does not
    // land in the
    // raise() accidentally. If single-stepping works, we will be spinning in
    // this loop. If
    // it doesn't, we'll land in the raise() call above.
    for (volatile unsigned i = 0; i < CPU_SETSIZE; ++i)
      ;
  }
}

struct ChildDeleter {
  ::pid_t pid;

  ~ChildDeleter() {
    int status;
    kill(pid, SIGKILL);            // Kill the child.
    waitpid(pid, &status, __WALL); // Pick up the remains.
  }
};

} // end anonymous namespace

bool impl::SingleStepWorkaroundNeeded() {
  // We shall spawn a child, and use it to verify the debug capabilities of the
  // cpu. We shall
  // iterate through the cpus, bind the child to each one in turn, and verify
  // that
  // single-stepping works on that cpu. A workaround is needed if we find at
  // least one broken
  // cpu.

  Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_THREAD));
  Error error;
  ::pid_t child_pid = fork();
  if (child_pid == -1) {
    if (log) {
      error.SetErrorToErrno();
      log->Printf("%s failed to fork(): %s", __FUNCTION__, error.AsCString());
    }
    return false;
  }
  if (child_pid == 0)
    Child();

  ChildDeleter child_deleter{child_pid};
  cpu_set_t available_cpus;
  if (sched_getaffinity(child_pid, sizeof available_cpus, &available_cpus) ==
      -1) {
    if (log) {
      error.SetErrorToErrno();
      log->Printf("%s failed to get available cpus: %s", __FUNCTION__,
                  error.AsCString());
    }
    return false;
  }

  int status;
  ::pid_t wpid = waitpid(child_pid, &status, __WALL);
  if (wpid != child_pid || !WIFSTOPPED(status)) {
    if (log) {
      error.SetErrorToErrno();
      log->Printf("%s waitpid() failed (status = %x): %s", __FUNCTION__, status,
                  error.AsCString());
    }
    return false;
  }

  unsigned cpu;
  for (cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &available_cpus))
      continue;

    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu, &cpus);
    if (sched_setaffinity(child_pid, sizeof cpus, &cpus) == -1) {
      if (log) {
        error.SetErrorToErrno();
        log->Printf("%s failed to switch to cpu %u: %s", __FUNCTION__, cpu,
                    error.AsCString());
      }
      continue;
    }

    int status;
    error = NativeProcessLinux::PtraceWrapper(PTRACE_SINGLESTEP, child_pid);
    if (error.Fail()) {
      if (log)
        log->Printf("%s single step failed: %s", __FUNCTION__,
                    error.AsCString());
      break;
    }

    wpid = waitpid(child_pid, &status, __WALL);
    if (wpid != child_pid || !WIFSTOPPED(status)) {
      if (log) {
        error.SetErrorToErrno();
        log->Printf("%s waitpid() failed (status = %x): %s", __FUNCTION__,
                    status, error.AsCString());
      }
      break;
    }
    if (WSTOPSIG(status) != SIGTRAP) {
      if (log)
        log->Printf("%s single stepping on cpu %d failed with status %x",
                    __FUNCTION__, cpu, status);
      break;
    }
  }

  // cpu is either the index of the first broken cpu, or CPU_SETSIZE.
  if (cpu == 0) {
    if (log)
      log->Printf("%s SINGLE STEPPING ON FIRST CPU IS NOT WORKING. DEBUGGING "
                  "LIKELY TO BE UNRELIABLE.",
                  __FUNCTION__);
    // No point in trying to fiddle with the affinities, just give it our best
    // shot and see how it goes.
    return false;
  }

  return cpu != CPU_SETSIZE;
}

#else // !arm64
bool impl::SingleStepWorkaroundNeeded() { return false; }
#endif
