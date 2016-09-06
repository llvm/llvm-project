//===-- llvm/Support/Threading.h - Control multithreading mode --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions for running LLVM in a multi-threaded
// environment.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREADING_H
#define LLVM_SUPPORT_THREADING_H

#include "llvm/Config/llvm-config.h" // for LLVM_ON_UNIX
#include "llvm/Support/Compiler.h"
#include <ciso646> // So we can check the C++ standard lib macros.
#include <functional>

// We use std::call_once on all Unix platforms except for NetBSD with
// libstdc++. That platform has a bug they are working to fix, and they'll
// remove the NetBSD checks once fixed.
#if defined(LLVM_ON_UNIX) &&                                                   \
    !(defined(__NetBSD__) && !defined(_LIBCPP_VERSION)) && !defined(__ppc__)
#define LLVM_THREADING_USE_STD_CALL_ONCE 1
#else
#define LLVM_THREADING_USE_STD_CALL_ONCE 0
#endif

#if LLVM_THREADING_USE_STD_CALL_ONCE
#include <mutex>
#else
#include "llvm/Support/Atomic.h"
#endif

namespace llvm {
  /// Returns true if LLVM is compiled with support for multi-threading, and
  /// false otherwise.
  bool llvm_is_multithreaded();

  /// llvm_execute_on_thread - Execute the given \p UserFn on a separate
  /// thread, passing it the provided \p UserData and waits for thread
  /// completion.
  ///
  /// This function does not guarantee that the code will actually be executed
  /// on a separate thread or honoring the requested stack size, but tries to do
  /// so where system support is available.
  ///
  /// \param UserFn - The callback to execute.
  /// \param UserData - An argument to pass to the callback function.
  /// \param RequestedStackSize - If non-zero, a requested size (in bytes) for
  /// the thread stack.
  void llvm_execute_on_thread(void (*UserFn)(void*), void *UserData,
                              unsigned RequestedStackSize = 0);

#if LLVM_THREADING_USE_STD_CALL_ONCE

  typedef std::once_flag once_flag;

  /// This macro is the only way you should define your once flag for LLVM's
  /// call_once.
#define LLVM_DEFINE_ONCE_FLAG(flag) static once_flag flag

#else

  enum InitStatus { Uninitialized = 0, Wait = 1, Done = 2 };
  typedef volatile sys::cas_flag once_flag;

  /// This macro is the only way you should define your once flag for LLVM's
  /// call_once.
#define LLVM_DEFINE_ONCE_FLAG(flag) static once_flag flag = Uninitialized

#endif

  /// \brief Execute the function specified as a parameter once.
  ///
  /// Typical usage:
  /// \code
  ///   void foo() {...};
  ///   ...
  ///   LLVM_DEFINE_ONCE_FLAG(flag);
  ///   call_once(flag, foo);
  /// \endcode
  ///
  /// \param flag Flag used for tracking whether or not this has run.
  /// \param F Function to call once.
  template <typename Function, typename... Args>
  void call_once(once_flag &flag, Function &&F, Args &&... ArgList) {
#if LLVM_THREADING_USE_STD_CALL_ONCE
    std::call_once(flag, std::forward<Function>(F),
                   std::forward<Args>(ArgList)...);
#else
    // For other platforms we use a generic (if brittle) version based on our
    // atomics.
    sys::cas_flag old_val = sys::CompareAndSwap(&flag, Wait, Uninitialized);
    if (old_val == Uninitialized) {
      std::forward<Function>(F)(std::forward<Args>(ArgList)...);
      sys::MemoryFence();
      TsanIgnoreWritesBegin();
      TsanHappensBefore(&flag);
      flag = Done;
      TsanIgnoreWritesEnd();
    } else {
      // Wait until any thread doing the call has finished.
      sys::cas_flag tmp = flag;
      sys::MemoryFence();
      while (tmp != Done) {
        tmp = flag;
        sys::MemoryFence();
      }
    }
    TsanHappensAfter(&flag);
#endif
  }
}

#endif
