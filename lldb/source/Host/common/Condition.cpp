//===-- Condition.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <errno.h>

#include "lldb/Host/Condition.h"
#include "lldb/Host/TimeValue.h"

using namespace lldb_private;

#ifndef _WIN32

//----------------------------------------------------------------------
// Default constructor
//
// The default constructor will initialize a new pthread condition
// and maintain the condition in the object state.
//----------------------------------------------------------------------
Condition::Condition() : m_condition() {
  ::pthread_cond_init(&m_condition, NULL);
}

//----------------------------------------------------------------------
// Destructor
//
// Destroys the pthread condition that the object owns.
//----------------------------------------------------------------------
Condition::~Condition() { ::pthread_cond_destroy(&m_condition); }

//----------------------------------------------------------------------
// Unblock all threads waiting for a condition variable
//----------------------------------------------------------------------
int Condition::Broadcast() { return ::pthread_cond_broadcast(&m_condition); }

//----------------------------------------------------------------------
// Unblocks one thread waiting for the condition variable
//----------------------------------------------------------------------
int Condition::Signal() { return ::pthread_cond_signal(&m_condition); }

//----------------------------------------------------------------------
// The Wait() function atomically blocks the current thread
// waiting on the owned condition variable, and unblocks the mutex
// specified by "mutex".  The waiting thread unblocks only after
// another thread calls Signal(), or Broadcast() with the same
// condition variable, or if "abstime" is valid (non-NULL) this
// function will return when the system time reaches the time
// specified in "abstime". If "abstime" is NULL this function will
// wait for an infinite amount of time for the condition variable
// to be signaled or broadcasted.
//
// The current thread re-acquires the lock on "mutex".
//----------------------------------------------------------------------
int Condition::Wait(Mutex &mutex, const TimeValue *abstime, bool *timed_out) {
  int err = 0;
  do {
    if (abstime && abstime->IsValid()) {
      struct timespec abstime_ts = abstime->GetAsTimeSpec();
      err =
          ::pthread_cond_timedwait(&m_condition, mutex.GetMutex(), &abstime_ts);
    } else
      err = ::pthread_cond_wait(&m_condition, mutex.GetMutex());
  } while (err == EINTR);

  if (timed_out != NULL) {
    if (err == ETIMEDOUT)
      *timed_out = true;
    else
      *timed_out = false;
  }

  return err;
}

#endif

//----------------------------------------------------------------------
// Get accessor to the pthread condition object
//----------------------------------------------------------------------
lldb::condition_t *Condition::GetCondition() { return &m_condition; }
