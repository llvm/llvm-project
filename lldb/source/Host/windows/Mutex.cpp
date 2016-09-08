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
#include "lldb/Host/windows/windows.h"

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

using namespace lldb_private;

//----------------------------------------------------------------------
// Default constructor.
//
// Creates a pthread mutex with no attributes.
//----------------------------------------------------------------------
Mutex::Mutex() : m_mutex() {
  m_mutex = static_cast<PCRITICAL_SECTION>(malloc(sizeof(CRITICAL_SECTION)));
  InitializeCriticalSection(static_cast<PCRITICAL_SECTION>(m_mutex));
}

//----------------------------------------------------------------------
// Default constructor.
//
// Creates a pthread mutex with "type" as the mutex type.
//----------------------------------------------------------------------
Mutex::Mutex(Mutex::Type type) : m_mutex() {
  m_mutex = static_cast<PCRITICAL_SECTION>(malloc(sizeof(CRITICAL_SECTION)));
  InitializeCriticalSection(static_cast<PCRITICAL_SECTION>(m_mutex));
}

//----------------------------------------------------------------------
// Destructor.
//
// Destroys the mutex owned by this object.
//----------------------------------------------------------------------
Mutex::~Mutex() {
  DeleteCriticalSection(static_cast<PCRITICAL_SECTION>(m_mutex));
  free(m_mutex);
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
            Host::GetCurrentProcessID(), Host::GetCurrentThreadID(), m_mutex);

  EnterCriticalSection(static_cast<PCRITICAL_SECTION>(m_mutex));
  return 0;
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
  return TryEnterCriticalSection(static_cast<PCRITICAL_SECTION>(m_mutex)) == 0;
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
  LeaveCriticalSection(static_cast<PCRITICAL_SECTION>(m_mutex));
  return 0;
}
