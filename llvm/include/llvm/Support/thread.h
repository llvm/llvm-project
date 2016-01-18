//===-- llvm/Support/thread.h - Wrapper for <thread> ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header is a wrapper for <thread> that works around problems with the
// MSVC headers when exceptions are disabled. It also provides llvm::thread,
// which is either a typedef of std::thread or a replacement that calls the
// function synchronously depending on the value of LLVM_ENABLE_THREADS.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_THREAD_H
#define LLVM_SUPPORT_THREAD_H

#include "llvm/Config/llvm-config.h"

#if LLVM_ENABLE_THREADS

#ifdef _MSC_VER
// concrt.h depends on eh.h for __uncaught_exception declaration
// even if we disable exceptions.
#include <eh.h>

// Suppress 'C++ exception handler used, but unwind semantics are not enabled.'
#pragma warning(push)
#pragma warning(disable:4530)
#endif

#include <thread>

#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace llvm {
typedef std::thread thread;
}

#else // !LLVM_ENABLE_THREADS

namespace llvm {

struct thread {
  thread() {}
  thread(thread &&other) {}
  template <class Function, class... Args>
  explicit thread(Function &&f, Args &&... args) {
    f(std::forward<Args>(args)...);
  }
  thread(const thread &) = delete;

  void join() {}
};

}

#endif // LLVM_ENABLE_THREADS

#endif
