//===- llvm/System/Mutex.h - Mutex Operating System Concept -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::Mutex class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_MUTEX_H
#define LLVM_SYSTEM_MUTEX_H

namespace llvm
{
  namespace sys
  {
    /// @brief Platform agnostic Mutex class.
    class Mutex
    {
    /// @name Constructors
    /// @{
    public:

      /// Initializes the lock but doesn't acquire it. if \p recursive is set
      /// to false, the lock will not be recursive which makes it cheaper but
      /// also more likely to deadlock (same thread can't acquire more than
      /// once).
      /// @brief Default Constructor.
      Mutex ( bool recursive = true );

      /// Releases and removes the lock
      /// @brief Destructor
      ~Mutex ( void );

    /// @}
    /// @name Methods
    /// @{
    public:

      /// Attempts to unconditionally acquire the lock. If the lock is held by
      /// another thread, this method will wait until it can acquire the lock.
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally acquire the lock.
      bool acquire();

      /// Attempts to release the lock. If the lock is held by the current
      /// thread, the lock is released allowing other threads to acquire the
      /// lock.
      /// @returns false if any kind of error occurs, true otherwise.
      /// @brief Unconditionally release the lock.
      bool release(void);

      /// Attempts to acquire the lock without blocking. If the lock is not
      /// available, this function returns false quickly (without blocking). If
      /// the lock is available, it is acquired.
      /// @returns false if any kind of error occurs or the lock is not
      /// available, true otherwise.
      /// @brief Try to acquire the lock.
      bool tryacquire();

    //@}
    /// @name Platform Dependent Data
    /// @{
    private:
#ifdef ENABLE_THREADS
      void* data_; ///< We don't know what the data will be
#endif

    /// @}
    /// @name Do Not Implement
    /// @{
    private:
      Mutex(const Mutex & original);
      void operator=(const Mutex &);
    /// @}
    };
  }
}

#endif
