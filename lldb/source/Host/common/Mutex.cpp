//===-- Mutex.cpp -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Mutex.h"
#include "lldb/Host/Host.h"

#ifndef _WIN32
#include <pthread.h>
#endif
#include <stdio.h>
#include <string.h>

#if 0
// This logging is way too verbose to enable even for a log channel. 
// This logging can be enabled by changing the "#if 0", but should be
// reverted prior to checking in.
#include <cstdio>
#define DEBUG_LOG(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// Enable extra mutex error checking
#if 0 // LLDB_CONFIGURATION_DEBUG
#define ENABLE_MUTEX_ERROR_CHECKING 1
#include <inttypes.h>
#endif

#if ENABLE_MUTEX_ERROR_CHECKING
#include <set>

enum MutexAction {
  eMutexActionInitialized,
  eMutexActionDestroyed,
  eMutexActionAssertInitialized
};

static bool error_check_mutex(pthread_mutex_t *m, MutexAction action) {
  typedef std::set<pthread_mutex_t *> mutex_set;
  static pthread_mutex_t g_mutex_set_mutex = PTHREAD_MUTEX_INITIALIZER;
  static mutex_set g_initialized_mutex_set;
  static mutex_set g_destroyed_mutex_set;

  bool success = true;
  int err;
  // Manually call lock so we don't to any of this error checking
  err = ::pthread_mutex_lock(&g_mutex_set_mutex);
  assert(err == 0);
  switch (action) {
  case eMutexActionInitialized:
    // Make sure this isn't already in our initialized mutex set...
    assert(g_initialized_mutex_set.find(m) == g_initialized_mutex_set.end());
    // Remove this from the destroyed set in case it was ever in there
    g_destroyed_mutex_set.erase(m);
    // Add the mutex to the initialized set
    g_initialized_mutex_set.insert(m);
    break;

  case eMutexActionDestroyed:
    // Make sure this isn't already in our destroyed mutex set...
    assert(g_destroyed_mutex_set.find(m) == g_destroyed_mutex_set.end());
    // Remove this from the initialized so we can put it into the destroyed set
    g_initialized_mutex_set.erase(m);
    // Add the mutex to the destroyed set
    g_destroyed_mutex_set.insert(m);
    break;
  case eMutexActionAssertInitialized:
    // This function will return true if "m" is in the initialized mutex set
    success = g_initialized_mutex_set.find(m) != g_initialized_mutex_set.end();
    assert(success);
    break;
  }
  // Manually call unlock so we don't to any of this error checking
  err = ::pthread_mutex_unlock(&g_mutex_set_mutex);
  assert(err == 0);
  return success;
}

#endif

using namespace lldb_private;

//----------------------------------------------------------------------
// Default constructor.
//
// This will create a scoped mutex locking object that doesn't have
// a mutex to lock. One will need to be provided using the Reset()
// method.
//----------------------------------------------------------------------
Mutex::Locker::Locker() : m_mutex_ptr(NULL) {}

//----------------------------------------------------------------------
// Constructor with a Mutex object.
//
// This will create a scoped mutex locking object that extracts the
// mutex owned by "m" and locks it.
//----------------------------------------------------------------------
Mutex::Locker::Locker(Mutex &m) : m_mutex_ptr(NULL) { Lock(m); }

//----------------------------------------------------------------------
// Constructor with a Mutex object pointer.
//
// This will create a scoped mutex locking object that extracts the
// mutex owned by "m" and locks it.
//----------------------------------------------------------------------
Mutex::Locker::Locker(Mutex *m) : m_mutex_ptr(NULL) {
  if (m)
    Lock(m);
}

//----------------------------------------------------------------------
// Destructor
//
// Unlocks any owned mutex object (if it is valid).
//----------------------------------------------------------------------
Mutex::Locker::~Locker() { Unlock(); }

//----------------------------------------------------------------------
// Unlock the current mutex in this object (if this owns a valid
// mutex) and lock the new "mutex" object if it is non-NULL.
//----------------------------------------------------------------------
void Mutex::Locker::Lock(Mutex &mutex) {
  // We already have this mutex locked or both are NULL...
  if (m_mutex_ptr == &mutex)
    return;

  Unlock();

  m_mutex_ptr = &mutex;
  m_mutex_ptr->Lock();
}

void Mutex::Locker::Unlock() {
  if (m_mutex_ptr) {
    m_mutex_ptr->Unlock();
    m_mutex_ptr = NULL;
  }
}

bool Mutex::Locker::TryLock(Mutex &mutex, const char *failure_message) {
  // We already have this mutex locked!
  if (m_mutex_ptr == &mutex)
    return true;

  Unlock();

  if (mutex.TryLock(failure_message) == 0)
    m_mutex_ptr = &mutex;

  return m_mutex_ptr != NULL;
}

#ifndef _WIN32

//----------------------------------------------------------------------
// Default constructor.
//
// Creates a pthread mutex with no attributes.
//----------------------------------------------------------------------
Mutex::Mutex() : m_mutex() {
  int err;
  err = ::pthread_mutex_init(&m_mutex, NULL);
#if ENABLE_MUTEX_ERROR_CHECKING
  if (err == 0)
    error_check_mutex(&m_mutex, eMutexActionInitialized);
#endif
  assert(err == 0);
}

//----------------------------------------------------------------------
// Default constructor.
//
// Creates a pthread mutex with "type" as the mutex type.
//----------------------------------------------------------------------
Mutex::Mutex(Mutex::Type type) : m_mutex() {
  int err;
  ::pthread_mutexattr_t attr;
  err = ::pthread_mutexattr_init(&attr);
  assert(err == 0);
  switch (type) {
  case eMutexTypeNormal:
#if ENABLE_MUTEX_ERROR_CHECKING
    err = ::pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
#else
    err = ::pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_NORMAL);
#endif
    break;

  case eMutexTypeRecursive:
    err = ::pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    break;
  }
  assert(err == 0);
  err = ::pthread_mutex_init(&m_mutex, &attr);
#if ENABLE_MUTEX_ERROR_CHECKING
  if (err == 0)
    error_check_mutex(&m_mutex, eMutexActionInitialized);
#endif
  assert(err == 0);
  err = ::pthread_mutexattr_destroy(&attr);
  assert(err == 0);
}

//----------------------------------------------------------------------
// Destructor.
//
// Destroys the mutex owned by this object.
//----------------------------------------------------------------------
Mutex::~Mutex() {
#if ENABLE_MUTEX_ERROR_CHECKING
  int err = ::pthread_mutex_destroy(&m_mutex);
  assert(err == 0);
  if (err == 0)
    error_check_mutex(&m_mutex, eMutexActionDestroyed);
  else {
    Host::SetCrashDescriptionWithFormat(
        "%s error: pthread_mutex_destroy() => err = %i (%s)",
        __PRETTY_FUNCTION__, err, strerror(err));
    assert(err == 0);
  }
  memset(&m_mutex, '\xba', sizeof(m_mutex));
#else
  ::pthread_mutex_destroy(&m_mutex);
#endif
}

//----------------------------------------------------------------------
// Locks the mutex owned by this object, if the mutex is already
// locked, the calling thread will block until the mutex becomes
// available.
//
// RETURNS
//  The error code from the pthread_mutex_lock() function call.
//----------------------------------------------------------------------
int Mutex::Lock() {
  DEBUG_LOG("[%4.4" PRIx64 "/%4.4" PRIx64 "] pthread_mutex_lock (%p)...\n",
            Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), &m_mutex);

#if ENABLE_MUTEX_ERROR_CHECKING
  error_check_mutex(&m_mutex, eMutexActionAssertInitialized);
#endif

  int err = ::pthread_mutex_lock(&m_mutex);

#if ENABLE_MUTEX_ERROR_CHECKING
  if (err) {
    Host::SetCrashDescriptionWithFormat(
        "%s error: pthread_mutex_lock(%p) => err = %i (%s)",
        __PRETTY_FUNCTION__, &m_mutex, err, strerror(err));
    assert(err == 0);
  }
#endif
  DEBUG_LOG("[%4.4" PRIx64 "/%4.4" PRIx64 "] pthread_mutex_lock (%p) => %i\n",
            Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), &m_mutex,
            err);
  return err;
}

//----------------------------------------------------------------------
// Attempts to lock the mutex owned by this object without blocking.
// If the mutex is already locked, TryLock() will not block waiting
// for the mutex, but will return an error condition.
//
// RETURNS
//  The error code from the pthread_mutex_trylock() function call.
//----------------------------------------------------------------------
int Mutex::TryLock(const char *failure_message) {
#if ENABLE_MUTEX_ERROR_CHECKING
  error_check_mutex(&m_mutex, eMutexActionAssertInitialized);
#endif

  int err = ::pthread_mutex_trylock(&m_mutex);
  DEBUG_LOG(
      "[%4.4" PRIx64 "/%4.4" PRIx64 "] pthread_mutex_trylock (%p) => %i\n",
      Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), &m_mutex, err);
  return err;
}

//----------------------------------------------------------------------
// If the current thread holds the lock on the owned mutex, then
// Unlock() will unlock the mutex. Calling Unlock() on this object
// that the calling thread does not hold will result in undefined
// behavior.
//
// RETURNS
//  The error code from the pthread_mutex_unlock() function call.
//----------------------------------------------------------------------
int Mutex::Unlock() {
#if ENABLE_MUTEX_ERROR_CHECKING
  error_check_mutex(&m_mutex, eMutexActionAssertInitialized);
#endif

  int err = ::pthread_mutex_unlock(&m_mutex);

#if ENABLE_MUTEX_ERROR_CHECKING
  if (err) {
    Host::SetCrashDescriptionWithFormat(
        "%s error: pthread_mutex_unlock(%p) => err = %i (%s)",
        __PRETTY_FUNCTION__, &m_mutex, err, strerror(err));
    assert(err == 0);
  }
#endif
  DEBUG_LOG("[%4.4" PRIx64 "/%4.4" PRIx64 "] pthread_mutex_unlock (%p) => %i\n",
            Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), &m_mutex,
            err);
  return err;
}

#endif

//----------------------------------------------------------------------
// Mutex get accessor.
//----------------------------------------------------------------------
lldb::mutex_t *Mutex::GetMutex() { return &m_mutex; }

#ifdef LLDB_CONFIGURATION_DEBUG
int TrackingMutex::Unlock() {
  if (!m_failure_message.empty())
    Host::SetCrashDescriptionWithFormat(
        "Unlocking lock (on thread %p) that thread: %p failed to get: %s",
        pthread_self(), m_thread_that_tried, m_failure_message.c_str());
  assert(m_failure_message.empty());
  return Mutex::Unlock();
}

int LoggingMutex::Lock() {
  printf("locking mutex %p by [%4.4" PRIx64 "/%4.4" PRIx64 "]...", this,
         Host::GetCurrentProcessID(), Host::GetCurrentThreadID());
  int x = Mutex::Lock();
  m_locked = true;
  printf("%d\n", x);
  return x;
}

int LoggingMutex::Unlock() {
  printf("unlocking mutex %p by [%4.4" PRIx64 "/%4.4" PRIx64 "]...", this,
         Host::GetCurrentProcessID(), Host::GetCurrentThreadID());
  int x = Mutex::Unlock();
  m_locked = false;
  printf("%d\n", x);
  return x;
}

int LoggingMutex::TryLock(const char *failure_message) {
  printf("trylocking mutex %p by [%4.4" PRIx64 "/%4.4" PRIx64 "]...", this,
         Host::GetCurrentProcessID(), Host::GetCurrentThreadID());
  int x = Mutex::TryLock(failure_message);
  if (x == 0)
    m_locked = true;
  printf("%d\n", x);
  return x;
}

#endif
