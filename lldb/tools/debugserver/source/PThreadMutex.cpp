//===-- PThreadMutex.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 12/9/08.
//
//===----------------------------------------------------------------------===//

#include "PThreadMutex.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "DNBTimer.h"

#if defined(DEBUG_PTHREAD_MUTEX_DEADLOCKS)

PThreadMutex::Locker::Locker(PThreadMutex &m, const char *function,
                             const char *file, const int line)
    : m_pMutex(m.Mutex()), m_function(function), m_file(file), m_line(line),
      m_lock_time(0) {
  Lock();
}

PThreadMutex::Locker::Locker(PThreadMutex *m, const char *function,
                             const char *file, const int line)
    : m_pMutex(m ? m->Mutex() : NULL), m_function(function), m_file(file),
      m_line(line), m_lock_time(0) {
  Lock();
}

PThreadMutex::Locker::Locker(pthread_mutex_t *mutex, const char *function,
                             const char *file, const int line)
    : m_pMutex(mutex), m_function(function), m_file(file), m_line(line),
      m_lock_time(0) {
  Lock();
}

PThreadMutex::Locker::~Locker() { Unlock(); }

void PThreadMutex::Locker::Lock() {
  if (m_pMutex) {
    m_lock_time = DNBTimer::GetTimeOfDay();
    if (::pthread_mutex_trylock(m_pMutex) != 0) {
      fprintf(stdout, "::pthread_mutex_trylock (%8.8p) mutex is locked "
                      "(function %s in %s:%i), waiting...\n",
              m_pMutex, m_function, m_file, m_line);
      ::pthread_mutex_lock(m_pMutex);
      fprintf(stdout, "::pthread_mutex_lock (%8.8p) succeeded after %6llu "
                      "usecs (function %s in %s:%i)\n",
              m_pMutex, DNBTimer::GetTimeOfDay() - m_lock_time, m_function,
              m_file, m_line);
    }
  }
}

void PThreadMutex::Locker::Unlock() {
  fprintf(stdout, "::pthread_mutex_unlock (%8.8p) had lock for %6llu usecs in "
                  "%s in %s:%i\n",
          m_pMutex, DNBTimer::GetTimeOfDay() - m_lock_time, m_function, m_file,
          m_line);
  ::pthread_mutex_unlock(m_pMutex);
}

#endif
