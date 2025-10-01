//===- IOSandbox.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IOSANDBOX_H
#define LLVM_SUPPORT_IOSANDBOX_H

// Always enable IO sandboxing in debug/assert builds for development,
// but allow enablement even for release/no-assert builds for production.
#if !defined(NDEBUG) || defined(LLVM_ENABLE_IO_SANDBOX)

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"

namespace llvm::sys::sandbox {
inline thread_local bool Enabled = false;
inline SaveAndRestore<bool> scopedEnable() { return {Enabled, true}; }
inline SaveAndRestore<bool> scopedDisable() { return {Enabled, false}; }
inline void violationIfEnabled() {
  if (Enabled)
    reportFatalInternalError("IO sandbox violation");
}
} // namespace llvm::sys::sandbox

#else

namespace llvm::sys::sandbox {
inline int scopedEnable() {}
inline int scopedDisable() {}
inline void violationIfEnabled() {}
} // namespace llvm::sys::sandbox

#endif

namespace llvm::sys::sandbox {
/// Facility for seamlessly interposing function calls and sandbox enforcement.
/// This is intended for creating static functors like so:
///
///   // before
///   #include <unistd.h>
///   namespace x {
///     void perform_read() { read(); } // not sandboxed
///   }
///
///   // after
///   #include <unistd.h>
///   namespace x {
///     static constexpr auto read = llvm::sys::sandbox::interpose(::read);
///     void perform_read() { read(); } // sandboxed
///   }
template <class FnTy> struct Interposed;

template <class RetTy, class... ArgTy> struct Interposed<RetTy (*)(ArgTy...)> {
  RetTy (*Fn)(ArgTy...);

  RetTy operator()(ArgTy... Arg) const {
    violationIfEnabled();
    return Fn(std::forward<ArgTy>(Arg)...);
  }
};

template <class RetTy, class... ArgTy>
struct Interposed<RetTy (*)(ArgTy..., ...)> {
  RetTy (*Fn)(ArgTy..., ...);

  template <class... CVarArgTy>
  RetTy operator()(ArgTy... Arg, CVarArgTy... CVarArg) const {
    violationIfEnabled();
    return Fn(std::forward<ArgTy>(Arg)..., std::forward<CVarArgTy>(CVarArg)...);
  }
};

template <class FnTy> constexpr auto interpose(FnTy Fn) {
  return Interposed<FnTy>{Fn};
}
} // namespace llvm::sys::sandbox

#endif
