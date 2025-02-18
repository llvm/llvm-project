//===-- MainLoopPosix.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/MainLoopPosix.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Utility/Status.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Errno.h"
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <vector>

// Multiplexing is implemented using kqueue on systems that support it (BSD
// variants including OSX). On linux we use ppoll.

#if HAVE_SYS_EVENT_H
#include <sys/event.h>
#else
#include <poll.h>
#endif

using namespace lldb;
using namespace lldb_private;

namespace {
struct GlobalSignalInfo {
  sig_atomic_t pipe_fd = -1;
  static_assert(sizeof(sig_atomic_t) >= sizeof(int),
                "Type too small for a file descriptor");
  sig_atomic_t flag = 0;
};
} // namespace
static GlobalSignalInfo g_signal_info[NSIG];

static void SignalHandler(int signo, siginfo_t *info, void *) {
  assert(signo < NSIG);

  // Set the flag before writing to the pipe!
  g_signal_info[signo].flag = 1;

  int fd = g_signal_info[signo].pipe_fd;
  if (fd < 0) {
    // This can happen with the following (unlikely) sequence of events:
    // 1. Thread 1 gets a signal, starts running the signal handler
    // 2. Thread 2 unregisters the signal handler, setting pipe_fd to -1
    // 3. Signal handler on thread 1 reads -1 out of pipe_fd
    // In this case, we can just ignore the signal because we're no longer
    // interested in it.
    return;
  }

  // Write a(ny) character to the pipe to wake up from the poll syscall.
  char c = '.';
  ssize_t bytes_written = llvm::sys::RetryAfterSignal(-1, ::write, fd, &c, 1);
  // We can safely ignore EAGAIN (pipe full), as that means poll will definitely
  // return.
  assert(bytes_written == 1 || (bytes_written == -1 && errno == EAGAIN));
  (void)bytes_written;
}

class ToTimeSpec {
public:
  explicit ToTimeSpec(std::optional<MainLoopPosix::TimePoint> point) {
    using namespace std::chrono;

    if (!point) {
      m_ts_ptr = nullptr;
      return;
    }
    nanoseconds dur = std::max(*point - steady_clock::now(), nanoseconds(0));
    m_ts_ptr = &m_ts;
    m_ts.tv_sec = duration_cast<seconds>(dur).count();
    m_ts.tv_nsec = (dur % seconds(1)).count();
  }
  ToTimeSpec(const ToTimeSpec &) = delete;
  ToTimeSpec &operator=(const ToTimeSpec &) = delete;

  operator struct timespec *() { return m_ts_ptr; }

private:
  struct timespec m_ts;
  struct timespec *m_ts_ptr;
};

class MainLoopPosix::RunImpl {
public:
  RunImpl(MainLoopPosix &loop);
  ~RunImpl() = default;

  Status Poll();

  void ProcessReadEvents();

private:
  MainLoopPosix &loop;

#if HAVE_SYS_EVENT_H
  std::vector<struct kevent> in_events;
  struct kevent out_events[4];
  int num_events = -1;

#else
  std::vector<struct pollfd> read_fds;
#endif
};

#if HAVE_SYS_EVENT_H
MainLoopPosix::RunImpl::RunImpl(MainLoopPosix &loop) : loop(loop) {
  in_events.reserve(loop.m_read_fds.size());
}

Status MainLoopPosix::RunImpl::Poll() {
  in_events.resize(loop.m_read_fds.size());
  unsigned i = 0;
  for (auto &fd : loop.m_read_fds)
    EV_SET(&in_events[i++], fd.first, EVFILT_READ, EV_ADD, 0, 0, 0);

  num_events =
      kevent(loop.m_kqueue, in_events.data(), in_events.size(), out_events,
             std::size(out_events), ToTimeSpec(loop.GetNextWakeupTime()));

  if (num_events < 0) {
    if (errno == EINTR) {
      // in case of EINTR, let the main loop run one iteration
      // we need to zero num_events to avoid assertions failing
      num_events = 0;
    } else
      return Status(errno, eErrorTypePOSIX);
  }
  return Status();
}

void MainLoopPosix::RunImpl::ProcessReadEvents() {
  assert(num_events >= 0);
  for (int i = 0; i < num_events; ++i) {
    if (loop.m_terminate_request)
      return;
    switch (out_events[i].filter) {
    case EVFILT_READ:
      loop.ProcessReadObject(out_events[i].ident);
      break;
    default:
      llvm_unreachable("Unknown event");
    }
  }
}
#else
MainLoopPosix::RunImpl::RunImpl(MainLoopPosix &loop) : loop(loop) {
  read_fds.reserve(loop.m_read_fds.size());
}

static int StartPoll(llvm::MutableArrayRef<struct pollfd> fds,
                     std::optional<MainLoopPosix::TimePoint> point) {
#if HAVE_PPOLL
  return ppoll(fds.data(), fds.size(), ToTimeSpec(point),
               /*sigmask=*/nullptr);
#else
  using namespace std::chrono;
  int timeout = -1;
  if (point) {
    nanoseconds dur = std::max(*point - steady_clock::now(), nanoseconds(0));
    timeout = ceil<milliseconds>(dur).count();
  }
  return poll(fds.data(), fds.size(), timeout);
#endif
}

Status MainLoopPosix::RunImpl::Poll() {
  read_fds.clear();

  for (const auto &fd : loop.m_read_fds) {
    struct pollfd pfd;
    pfd.fd = fd.first;
    pfd.events = POLLIN;
    pfd.revents = 0;
    read_fds.push_back(pfd);
  }
  int ready = StartPoll(read_fds, loop.GetNextWakeupTime());

  if (ready == -1 && errno != EINTR)
    return Status(errno, eErrorTypePOSIX);

  return Status();
}

void MainLoopPosix::RunImpl::ProcessReadEvents() {
  for (const auto &fd : read_fds) {
    if ((fd.revents & (POLLIN | POLLHUP)) == 0)
      continue;
    IOObject::WaitableHandle handle = fd.fd;
    if (loop.m_terminate_request)
      return;

    loop.ProcessReadObject(handle);
  }
}
#endif

MainLoopPosix::MainLoopPosix() {
  Status error = m_interrupt_pipe.CreateNew(/*child_process_inherit=*/false);
  assert(error.Success());

  // Make the write end of the pipe non-blocking.
  int result = fcntl(m_interrupt_pipe.GetWriteFileDescriptor(), F_SETFL,
                     fcntl(m_interrupt_pipe.GetWriteFileDescriptor(), F_GETFL) |
                         O_NONBLOCK);
  assert(result == 0);
  UNUSED_IF_ASSERT_DISABLED(result);

  const int interrupt_pipe_fd = m_interrupt_pipe.GetReadFileDescriptor();
  m_read_fds.insert(
      {interrupt_pipe_fd, [interrupt_pipe_fd](MainLoopBase &loop) {
         char c;
         ssize_t bytes_read =
             llvm::sys::RetryAfterSignal(-1, ::read, interrupt_pipe_fd, &c, 1);
         assert(bytes_read == 1);
         UNUSED_IF_ASSERT_DISABLED(bytes_read);
         // NB: This implicitly causes another loop iteration
         // and therefore the execution of pending callbacks.
       }});
#if HAVE_SYS_EVENT_H
  m_kqueue = kqueue();
  assert(m_kqueue >= 0);
#endif
}

MainLoopPosix::~MainLoopPosix() {
#if HAVE_SYS_EVENT_H
  close(m_kqueue);
#endif
  m_read_fds.erase(m_interrupt_pipe.GetReadFileDescriptor());
  m_interrupt_pipe.Close();
  assert(m_read_fds.size() == 0);
  assert(m_signals.size() == 0);
}

MainLoopPosix::ReadHandleUP
MainLoopPosix::RegisterReadObject(const IOObjectSP &object_sp,
                                  const Callback &callback, Status &error) {
  if (!object_sp || !object_sp->IsValid()) {
    error = Status::FromErrorString("IO object is not valid.");
    return nullptr;
  }

  const bool inserted =
      m_read_fds.insert({object_sp->GetWaitableHandle(), callback}).second;
  if (!inserted) {
    error = Status::FromErrorStringWithFormat(
        "File descriptor %d already monitored.",
        object_sp->GetWaitableHandle());
    return nullptr;
  }

  return CreateReadHandle(object_sp);
}

// We shall block the signal, then install the signal handler. The signal will
// be unblocked in the Run() function to check for signal delivery.
MainLoopPosix::SignalHandleUP
MainLoopPosix::RegisterSignal(int signo, const Callback &callback,
                              Status &error) {
  auto signal_it = m_signals.find(signo);
  if (signal_it != m_signals.end()) {
    auto callback_it = signal_it->second.callbacks.insert(
        signal_it->second.callbacks.end(), callback);
    return SignalHandleUP(new SignalHandle(*this, signo, callback_it));
  }

  SignalInfo info;
  info.callbacks.push_back(callback);
  struct sigaction new_action;
  new_action.sa_sigaction = &SignalHandler;
  new_action.sa_flags = SA_SIGINFO;
  sigemptyset(&new_action.sa_mask);
  sigaddset(&new_action.sa_mask, signo);
  sigset_t old_set;

  // Set signal info before installing the signal handler!
  g_signal_info[signo].pipe_fd = m_interrupt_pipe.GetWriteFileDescriptor();
  g_signal_info[signo].flag = 0;

  int ret = sigaction(signo, &new_action, &info.old_action);
  UNUSED_IF_ASSERT_DISABLED(ret);
  assert(ret == 0 && "sigaction failed");

  ret = pthread_sigmask(SIG_UNBLOCK, &new_action.sa_mask, &old_set);
  assert(ret == 0 && "pthread_sigmask failed");
  info.was_blocked = sigismember(&old_set, signo);
  auto insert_ret = m_signals.insert({signo, info});

  return SignalHandleUP(new SignalHandle(
      *this, signo, insert_ret.first->second.callbacks.begin()));
}

void MainLoopPosix::UnregisterReadObject(IOObject::WaitableHandle handle) {
  bool erased = m_read_fds.erase(handle);
  UNUSED_IF_ASSERT_DISABLED(erased);
  assert(erased);
}

void MainLoopPosix::UnregisterSignal(
    int signo, std::list<Callback>::iterator callback_it) {
  auto it = m_signals.find(signo);
  assert(it != m_signals.end());

  it->second.callbacks.erase(callback_it);
  // Do not remove the signal handler unless all callbacks have been erased.
  if (!it->second.callbacks.empty())
    return;

  sigaction(signo, &it->second.old_action, nullptr);

  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, signo);
  int ret = pthread_sigmask(it->second.was_blocked ? SIG_BLOCK : SIG_UNBLOCK,
                            &set, nullptr);
  assert(ret == 0);
  UNUSED_IF_ASSERT_DISABLED(ret);

  m_signals.erase(it);
  g_signal_info[signo] = {};
}

Status MainLoopPosix::Run() {
  m_terminate_request = false;

  Status error;
  RunImpl impl(*this);

  while (!m_terminate_request) {
    error = impl.Poll();
    if (error.Fail())
      return error;

    impl.ProcessReadEvents();

    ProcessSignals();

    m_interrupting = false;
    ProcessCallbacks();
  }
  return Status();
}

void MainLoopPosix::ProcessReadObject(IOObject::WaitableHandle handle) {
  auto it = m_read_fds.find(handle);
  if (it != m_read_fds.end())
    it->second(*this); // Do the work
}

void MainLoopPosix::ProcessSignals() {
  std::vector<int> signals;
  for (const auto &entry : m_signals)
    if (g_signal_info[entry.first].flag != 0)
      signals.push_back(entry.first);

  for (const auto &signal : signals) {
    if (m_terminate_request)
      return;

    g_signal_info[signal].flag = 0;
    ProcessSignal(signal);
  }
}

void MainLoopPosix::ProcessSignal(int signo) {
  auto it = m_signals.find(signo);
  if (it != m_signals.end()) {
    // The callback may actually register/unregister signal handlers,
    // so we need to create a copy first.
    llvm::SmallVector<Callback, 4> callbacks_to_run{
        it->second.callbacks.begin(), it->second.callbacks.end()};
    for (auto &x : callbacks_to_run)
      x(*this); // Do the work
  }
}

void MainLoopPosix::Interrupt() {
  if (m_interrupting.exchange(true))
    return;

  char c = '.';
  size_t bytes_written;
  Status error = m_interrupt_pipe.Write(&c, 1, bytes_written);
  assert(error.Success());
  UNUSED_IF_ASSERT_DISABLED(error);
  assert(bytes_written == 1);
}
