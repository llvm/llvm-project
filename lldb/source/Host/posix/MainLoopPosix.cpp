//===-- MainLoopPosix.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/MainLoopPosix.h"
#include "lldb/Utility/Error.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <csignal>
#include <sys/select.h>
#include <vector>

using namespace lldb;
using namespace lldb_private;

static sig_atomic_t g_signal_flags[NSIG];

static void SignalHandler(int signo, siginfo_t *info, void *) {
  assert(signo < NSIG);
  g_signal_flags[signo] = 1;
}

MainLoopPosix::~MainLoopPosix() {
  assert(m_read_fds.size() == 0);
  assert(m_signals.size() == 0);
}

MainLoopPosix::ReadHandleUP
MainLoopPosix::RegisterReadObject(const IOObjectSP &object_sp,
                                  const Callback &callback, Error &error) {
  if (!object_sp || !object_sp->IsValid()) {
    error.SetErrorString("IO object is not valid.");
    return nullptr;
  }

  const bool inserted =
      m_read_fds.insert({object_sp->GetWaitableHandle(), callback}).second;
  if (!inserted) {
    error.SetErrorStringWithFormat("File descriptor %d already monitored.",
                                   object_sp->GetWaitableHandle());
    return nullptr;
  }

  return CreateReadHandle(object_sp);
}

// We shall block the signal, then install the signal handler. The signal will
// be unblocked in
// the Run() function to check for signal delivery.
MainLoopPosix::SignalHandleUP
MainLoopPosix::RegisterSignal(int signo, const Callback &callback,
                              Error &error) {
  if (m_signals.find(signo) != m_signals.end()) {
    error.SetErrorStringWithFormat("Signal %d already monitored.", signo);
    return nullptr;
  }

  SignalInfo info;
  info.callback = callback;
  struct sigaction new_action;
  new_action.sa_sigaction = &SignalHandler;
  new_action.sa_flags = SA_SIGINFO;
  sigemptyset(&new_action.sa_mask);
  sigaddset(&new_action.sa_mask, signo);

  sigset_t old_set;
  if (int ret = pthread_sigmask(SIG_BLOCK, &new_action.sa_mask, &old_set)) {
    error.SetErrorStringWithFormat("pthread_sigmask failed with error %d\n",
                                   ret);
    return nullptr;
  }

  info.was_blocked = sigismember(&old_set, signo);
  if (sigaction(signo, &new_action, &info.old_action) == -1) {
    error.SetErrorToErrno();
    if (!info.was_blocked)
      pthread_sigmask(SIG_UNBLOCK, &new_action.sa_mask, nullptr);
    return nullptr;
  }

  m_signals.insert({signo, info});
  g_signal_flags[signo] = 0;

  return SignalHandleUP(new SignalHandle(*this, signo));
}

void MainLoopPosix::UnregisterReadObject(IOObject::WaitableHandle handle) {
  bool erased = m_read_fds.erase(handle);
  UNUSED_IF_ASSERT_DISABLED(erased);
  assert(erased);
}

void MainLoopPosix::UnregisterSignal(int signo) {
  // We undo the actions of RegisterSignal on a best-effort basis.
  auto it = m_signals.find(signo);
  assert(it != m_signals.end());

  sigaction(signo, &it->second.old_action, nullptr);

  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, signo);
  pthread_sigmask(it->second.was_blocked ? SIG_BLOCK : SIG_UNBLOCK, &set,
                  nullptr);

  m_signals.erase(it);
}

Error MainLoopPosix::Run() {
  std::vector<int> signals;
  sigset_t sigmask;
  std::vector<int> read_fds;
  fd_set read_fd_set;
  m_terminate_request = false;

  // run until termination or until we run out of things to listen to
  while (!m_terminate_request && (!m_read_fds.empty() || !m_signals.empty())) {
    // To avoid problems with callbacks changing the things we're supposed to
    // listen to, we
    // will store the *real* list of events separately.
    signals.clear();
    read_fds.clear();
    FD_ZERO(&read_fd_set);
    int nfds = 0;

    if (int ret = pthread_sigmask(SIG_SETMASK, nullptr, &sigmask))
      return Error("pthread_sigmask failed with error %d\n", ret);

    for (const auto &fd : m_read_fds) {
      read_fds.push_back(fd.first);
      FD_SET(fd.first, &read_fd_set);
      nfds = std::max(nfds, fd.first + 1);
    }

    for (const auto &sig : m_signals) {
      signals.push_back(sig.first);
      sigdelset(&sigmask, sig.first);
    }

    if (pselect(nfds, &read_fd_set, nullptr, nullptr, nullptr, &sigmask) ==
            -1 &&
        errno != EINTR)
      return Error(errno, eErrorTypePOSIX);

    for (int sig : signals) {
      if (g_signal_flags[sig] == 0)
        continue; // No signal
      g_signal_flags[sig] = 0;

      auto it = m_signals.find(sig);
      if (it == m_signals.end())
        continue; // Signal must have gotten unregistered in the meantime

      it->second.callback(*this); // Do the work

      if (m_terminate_request)
        return Error();
    }

    for (int fd : read_fds) {
      if (!FD_ISSET(fd, &read_fd_set))
        continue; // Not ready

      auto it = m_read_fds.find(fd);
      if (it == m_read_fds.end())
        continue; // File descriptor must have gotten unregistered in the
                  // meantime

      it->second(*this); // Do the work

      if (m_terminate_request)
        return Error();
    }
  }
  return Error();
}
