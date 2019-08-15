//===- RWMutex.h - Reader/Writer Mutual Exclusion Lock ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys::RWMutex class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_RWMUTEX_H
#define LLVM_SUPPORT_RWMUTEX_H

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Threading.h"
#include <cassert>
#include <mutex>
#include <shared_mutex>

namespace llvm {
namespace sys {

    /// SmartMutex - An R/W mutex with a compile time constant parameter that
    /// indicates whether this mutex should become a no-op when we're not
    /// running in multithreaded mode.
    template<bool mt_only>
    class SmartRWMutex {
      // shared_mutex (C++17) is more efficient than shared_timed_mutex (C++14)
      // on Windows and always available on MSVC.
#if defined(_MSC_VER) || __cplusplus > 201402L
      std::shared_mutex impl;
#else
      // std::shared_timed_mutex is only availble on macOS 10.12 and later.
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&                  \
    defined(__MAC_10_12) &&                                                    \
    __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ < __MAC_10_12
      /// Platform agnostic RWMutex class.
      /// Platform agnostic RWMutex class.
      class RWMutexImpl {
        /// @name Constructors
        /// @{
      public:
        /// Initializes the lock but doesn't acquire it.
        /// Default Constructor.
        explicit RWMutexImpl();

        /// @}
        /// @name Do Not Implement
        /// @{
        RWMutexImpl(const RWMutexImpl &original) = delete;
        RWMutexImpl &operator=(const RWMutexImpl &) = delete;
        /// @}

        /// Releases and removes the lock
        /// Destructor
        ~RWMutexImpl();

        /// @}
        /// @name Methods
        /// @{
      public:
        /// Attempts to unconditionally acquire the lock in reader mode. If the
        /// lock is held by a writer, this method will wait until it can acquire
        /// the lock.
        /// @returns false if any kind of error occurs, true otherwise.
        /// Unconditionally acquire the lock in reader mode.
        bool reader_acquire();

        /// Attempts to release the lock in reader mode.
        /// @returns false if any kind of error occurs, true otherwise.
        /// Unconditionally release the lock in reader mode.
        bool reader_release();

        /// Attempts to unconditionally acquire the lock in reader mode. If the
        /// lock is held by any readers, this method will wait until it can
        /// acquire the lock.
        /// @returns false if any kind of error occurs, true otherwise.
        /// Unconditionally acquire the lock in writer mode.
        bool writer_acquire();

        /// Attempts to release the lock in writer mode.
        /// @returns false if any kind of error occurs, true otherwise.
        /// Unconditionally release the lock in write mode.
        bool writer_release();

        //@}
        /// @name Platform Dependent Data
        /// @{
      private:
#if defined(LLVM_ENABLE_THREADS) && LLVM_ENABLE_THREADS != 0
        void *data_ = nullptr; ///< We don't know what the data will be
#endif
      } impl;
#else
      std::shared_timed_mutex impl;
#endif
#endif
      unsigned readers = 0;
      unsigned writers = 0;

    public:
      bool lock_shared() {
        if (!mt_only || llvm_is_multithreaded()) {
          impl.lock_shared();
          return true;
        }

        // Single-threaded debugging code.  This would be racy in multithreaded
        // mode, but provides not sanity checks in single threaded mode.
        ++readers;
        return true;
      }

      bool unlock_shared() {
        if (!mt_only || llvm_is_multithreaded()) {
          impl.unlock_shared();
          return true;
        }

        // Single-threaded debugging code.  This would be racy in multithreaded
        // mode, but provides not sanity checks in single threaded mode.
        assert(readers > 0 && "Reader lock not acquired before release!");
        --readers;
        return true;
      }

      bool lock() {
        if (!mt_only || llvm_is_multithreaded()) {
          impl.lock();
          return true;
        }

        // Single-threaded debugging code.  This would be racy in multithreaded
        // mode, but provides not sanity checks in single threaded mode.
        assert(writers == 0 && "Writer lock already acquired!");
        ++writers;
        return true;
      }

      bool unlock() {
        if (!mt_only || llvm_is_multithreaded()) {
          impl.unlock();
          return true;
        }

        // Single-threaded debugging code.  This would be racy in multithreaded
        // mode, but provides not sanity checks in single threaded mode.
        assert(writers == 1 && "Writer lock not acquired before release!");
        --writers;
        return true;
      }
    };

    typedef SmartRWMutex<false> RWMutex;

    /// ScopedReader - RAII acquisition of a reader lock
    template<bool mt_only>
    using SmartScopedReader = const std::shared_lock<SmartRWMutex<mt_only>>;

    typedef SmartScopedReader<false> ScopedReader;

    /// ScopedWriter - RAII acquisition of a writer lock
    template<bool mt_only>
    using SmartScopedWriter = std::lock_guard<SmartRWMutex<mt_only>>;

    typedef SmartScopedWriter<false> ScopedWriter;

} // end namespace sys
} // end namespace llvm

#endif // LLVM_SUPPORT_RWMUTEX_H
