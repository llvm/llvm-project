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
#include "lldb/Host/windows/windows.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// Default constructor
//
// The default constructor will initialize a new pthread condition
// and maintain the condition in the object state.
//----------------------------------------------------------------------
Condition::Condition() : m_condition() {
  m_condition =
      static_cast<PCONDITION_VARIABLE>(malloc(sizeof(CONDITION_VARIABLE)));
  InitializeConditionVariable(static_cast<PCONDITION_VARIABLE>(m_condition));
}

//----------------------------------------------------------------------
// Destructor
//
// Destroys the pthread condition that the object owns.
//----------------------------------------------------------------------
Condition::~Condition() { free(m_condition); }

//----------------------------------------------------------------------
// Unblock all threads waiting for a condition variable
//----------------------------------------------------------------------
int Condition::Broadcast() {
  WakeAllConditionVariable(static_cast<PCONDITION_VARIABLE>(m_condition));
  return 0;
}

//----------------------------------------------------------------------
// Unblocks one thread waiting for the condition variable
//----------------------------------------------------------------------
int Condition::Signal() {
  WakeConditionVariable(static_cast<PCONDITION_VARIABLE>(m_condition));
  return 0;
}

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
  DWORD wait = INFINITE;
  if (abstime != NULL) {
    int wval = (*abstime - TimeValue::Now()) / 1000000;
    if (wval < 0)
      wval = 0;

    wait = wval;
  }

  int err = SleepConditionVariableCS(
      static_cast<PCONDITION_VARIABLE>(m_condition),
      static_cast<PCRITICAL_SECTION>(mutex.m_mutex), wait);

  if (timed_out != NULL) {
    if ((err == 0) && GetLastError() == ERROR_TIMEOUT)
      *timed_out = true;
    else
      *timed_out = false;
  }

  return err == 0;
}
