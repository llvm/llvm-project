//===-- ProcessRunLock.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/ProcessRunLock.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Threading.h"

#include <cassert>
#include <cstdint>

namespace lldb_private {

#ifndef _WIN32

ProcessRunLock::ProcessRunLock() {
  int err = ::pthread_rwlock_init(&m_rwlock, nullptr);
  (void)err;
}

ProcessRunLock::~ProcessRunLock() {
  int err = ::pthread_rwlock_destroy(&m_rwlock);
  (void)err;
}

bool ProcessRunLock::ReadTryLock() {
  ::pthread_rwlock_rdlock(&m_rwlock);
  if (!m_running) {
    // coverity[missing_unlock]
    return true;
  }
  ::pthread_rwlock_unlock(&m_rwlock);
  return false;
}

bool ProcessRunLock::ReadUnlock() {
  return ::pthread_rwlock_unlock(&m_rwlock) == 0;
}

bool ProcessRunLock::SetRunning() {
  ::pthread_rwlock_wrlock(&m_rwlock);
  bool was_stopped = !m_running;
  m_running = true;
  ::pthread_rwlock_unlock(&m_rwlock);
  return was_stopped;
}

bool ProcessRunLock::SetStopped() {
  ::pthread_rwlock_wrlock(&m_rwlock);
  bool was_running = m_running;
  m_running = false;
  ::pthread_rwlock_unlock(&m_rwlock);
  return was_running;
}

#endif // !_WIN32

ProcessRunLock::ProcessRunLocker::ProcessRunLocker(ProcessRunLocker &&other) {
  if (other.m_lock && other.m_thread != llvm::get_threadid())
    llvm::report_fatal_error(
        "ProcessRunLocker moved across threads while held");
  assert(!m_lock && "move-construct into a held ProcessRunLocker");
  m_lock = other.m_lock;
  m_thread = other.m_thread;
  other.m_lock = nullptr;
}

ProcessRunLock::ProcessRunLocker &
ProcessRunLock::ProcessRunLocker::operator=(ProcessRunLocker &&other) {
  if (this != &other) {
    if (other.m_lock && other.m_thread != llvm::get_threadid())
      llvm::report_fatal_error(
          "ProcessRunLocker move-assigned across threads while held");
    Unlock();
    m_lock = other.m_lock;
    m_thread = other.m_thread;
    other.m_lock = nullptr;
  }
  return *this;
}

bool ProcessRunLock::ProcessRunLocker::TryLock(ProcessRunLock *lock) {
  if (m_lock) {
    if (m_lock == lock)
      return true;
    Unlock();
  }
  if (!lock)
    return false;

  const uint64_t this_thread = llvm::get_threadid();

  // Fast path: already holding as a reader on this thread, bump the count.
  {
    std::lock_guard<std::mutex> guard(lock->m_recursion_mutex);
    auto it = lock->m_recursion.find(this_thread);
    if (it != lock->m_recursion.end()) {
      ++it->second;
      m_lock = lock;
      m_thread = this_thread;
      return true;
    }
  }

  // Acquire the rwlock with m_recursion_mutex released: ReadTryLock can
  // block waiting for a writer, and holding the recursion mutex through
  // that wait would stall fast-path bumps that only need the map.
  if (!lock->ReadTryLock())
    return false;

  std::lock_guard<std::mutex> guard(lock->m_recursion_mutex);
  lock->m_recursion[this_thread] = 1;
  m_lock = lock;
  m_thread = this_thread;
  return true;
}

void ProcessRunLock::ProcessRunLocker::Unlock() {
  if (!m_lock)
    return;

  const uint64_t this_thread = llvm::get_threadid();
  if (m_thread != this_thread)
    // pthread_rwlock_unlock from a different thread than the one that
    // called pthread_rwlock_rdlock is UB. The move ctor / operator= are
    // the only way a held locker can change object identity, and both
    // fatal on cross-thread transfer; this is the last-line check on
    // the destructor path in case a held locker escaped that trap.
    llvm::report_fatal_error(
        "ProcessRunLocker destroyed on a different thread while held");

  bool release_rwlock = false;
  {
    std::lock_guard<std::mutex> guard(m_lock->m_recursion_mutex);
    auto it = m_lock->m_recursion.find(this_thread);
    assert(
        it != m_lock->m_recursion.end() &&
        "ProcessRunLocker released without a matching TryLock on this thread");
    if (it == m_lock->m_recursion.end()) {
      m_lock = nullptr;
      return;
    }
    if (--it->second == 0) {
      m_lock->m_recursion.erase(it);
      release_rwlock = true;
    }
  }

  if (release_rwlock)
    m_lock->ReadUnlock();
  m_lock = nullptr;
}

} // namespace lldb_private
