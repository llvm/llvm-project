//===--------------------- KQueue.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "KQueue.h"

#ifdef LLDB_USE_KQUEUES

#include "lldb/Core/Error.h"

#include "Utility/TimeSpecTimeout.h"

using namespace lldb_private;

int KQueue::GetFD(bool can_create) {
  if (!IsValid() && can_create)
    m_fd = kqueue();
  return m_fd;
}

int KQueue::Close() {
  const int fd = m_fd;
  if (fd >= 0) {
    m_fd = -1;
    return close(fd);
  }
  return 0;
}

int KQueue::WaitForEvents(struct kevent *events, int num_events, Error &error,
                          uint32_t timeout_usec) {
  const int fd_kqueue = GetFD(false);
  if (fd_kqueue >= 0) {
    TimeSpecTimeout timeout;
    const struct timespec *timeout_ptr =
        timeout.SetRelativeTimeoutMircoSeconds32(timeout_usec);
    int result = ::kevent(fd_kqueue, NULL, 0, events, num_events, timeout_ptr);
    if (result == -1)
      error.SetErrorToErrno();
    else
      error.Clear();
    return result;
  } else {
    error.SetErrorString("invalid kqueue fd");
  }
  return 0;
}

bool KQueue::AddFDEvent(int fd, bool read, bool write, bool vnode) {
  const int fd_kqueue = GetFD(true);
  if (fd_kqueue >= 0) {
    struct kevent event;
    event.ident = fd;
    event.filter = 0;
    if (read)
      event.filter |= EVFILT_READ;
    if (write)
      event.filter |= EVFILT_WRITE;
    if (vnode)
      event.filter |= EVFILT_VNODE;
    event.flags = EV_ADD | EV_CLEAR;
    event.fflags = 0;
    event.data = 0;
    event.udata = NULL;
    int err = ::kevent(fd_kqueue, &event, 1, NULL, 0, NULL);
    return err == 0;
  }
  return false;
}

#endif
