//===- Mutex.cpp - Mutual Exclusion Lock ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the llvm::sys::Mutex class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"
#include "llvm/System/Mutex.h"

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only TRULY operating system
//===          independent code.
//===----------------------------------------------------------------------===//

#if !defined(ENABLE_THREADS) || ENABLE_THREADS == 0
// Define all methods as no-ops if threading is explicitly disabled
namespace llvm {
using namespace sys;
Mutex::Mutex( bool recursive) { }
Mutex::~Mutex() { }
bool Mutex::acquire() { return true; }
bool Mutex::release() { return true; }
bool Mutex::tryacquire() { return true; }
}
#else

#if defined(HAVE_PTHREAD_H) && defined(HAVE_PTHREAD_MUTEX_LOCK)

#include <cassert>
#include <pthread.h>
#include <stdlib.h>

namespace llvm {
using namespace sys;


// This variable is useful for situations where the pthread library has been
// compiled with weak linkage for its interface symbols. This allows the
// threading support to be turned off by simply not linking against -lpthread.
// In that situation, the value of pthread_mutex_init will be 0 and
// consequently pthread_enabled will be false. In such situations, all the
// pthread operations become no-ops and the functions all return false. If
// pthread_mutex_init does have an address, then mutex support is enabled.
// Note: all LLVM tools will link against -lpthread if its available since it
//       is configured into the LIBS variable.
// Note: this line of code generates a warning if pthread_mutex_init is not
//       declared with weak linkage. It's safe to ignore the warning.
static const bool pthread_enabled = true;

// Construct a Mutex using pthread calls
Mutex::Mutex( bool recursive)
  : data_(0)
{
  if (pthread_enabled)
  {
    // Declare the pthread_mutex data structures
    pthread_mutex_t* mutex =
      static_cast<pthread_mutex_t*>(malloc(sizeof(pthread_mutex_t)));
    pthread_mutexattr_t attr;

    // Initialize the mutex attributes
    int errorcode = pthread_mutexattr_init(&attr);
    assert(errorcode == 0);

    // Initialize the mutex as a recursive mutex, if requested, or normal
    // otherwise.
    int kind = ( recursive  ? PTHREAD_MUTEX_RECURSIVE : PTHREAD_MUTEX_NORMAL );
    errorcode = pthread_mutexattr_settype(&attr, kind);
    assert(errorcode == 0);

#if !defined(__FreeBSD__) && !defined(__OpenBSD__) && !defined(__NetBSD__) && !defined(__DragonFly__)
    // Make it a process local mutex
    errorcode = pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_PRIVATE);
#endif

    // Initialize the mutex
    errorcode = pthread_mutex_init(mutex, &attr);
    assert(errorcode == 0);

    // Destroy the attributes
    errorcode = pthread_mutexattr_destroy(&attr);
    assert(errorcode == 0);

    // Assign the data member
    data_ = mutex;
  }
}

// Destruct a Mutex
Mutex::~Mutex()
{
  if (pthread_enabled)
  {
    pthread_mutex_t* mutex = static_cast<pthread_mutex_t*>(data_);
    assert(mutex != 0);
    pthread_mutex_destroy(mutex);
    free(mutex);
  }
}

bool
Mutex::acquire()
{
  if (pthread_enabled)
  {
    pthread_mutex_t* mutex = static_cast<pthread_mutex_t*>(data_);
    assert(mutex != 0);

    int errorcode = pthread_mutex_lock(mutex);
    return errorcode == 0;
  }
  return false;
}

bool
Mutex::release()
{
  if (pthread_enabled)
  {
    pthread_mutex_t* mutex = static_cast<pthread_mutex_t*>(data_);
    assert(mutex != 0);

    int errorcode = pthread_mutex_unlock(mutex);
    return errorcode == 0;
  }
  return false;
}

bool
Mutex::tryacquire()
{
  if (pthread_enabled)
  {
    pthread_mutex_t* mutex = static_cast<pthread_mutex_t*>(data_);
    assert(mutex != 0);

    int errorcode = pthread_mutex_trylock(mutex);
    return errorcode == 0;
  }
  return false;
}

}

#elif defined(LLVM_ON_UNIX)
#include "Unix/Mutex.inc"
#elif defined( LLVM_ON_WIN32)
#include "Win32/Mutex.inc"
#else
#warning Neither LLVM_ON_UNIX nor LLVM_ON_WIN32 was set in System/Mutex.cpp
#endif
#endif

