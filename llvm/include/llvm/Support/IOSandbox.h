//===- IOSandbox.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_IOSANDBOX_H
#define LLVM_SUPPORT_IOSANDBOX_H

#if defined(LLVM_ENABLE_IO_SANDBOX) && LLVM_ENABLE_IO_SANDBOX

#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"

namespace llvm::sys::sandbox {
inline LLVM_THREAD_LOCAL bool Enabled = false;
struct ScopedSetting {
  SaveAndRestore<bool> Impl;
};
inline ScopedSetting scopedEnable() { return {{Enabled, true}}; }
inline ScopedSetting scopedDisable() { return {{Enabled, false}}; }
inline void violationIfEnabled() {
  if (Enabled)
    reportFatalInternalError("IO sandbox violation");
}
} // namespace llvm::sys::sandbox

#else

namespace llvm::sys::sandbox {
struct [[maybe_unused]] ScopedSetting {};
inline ScopedSetting scopedEnable() { return {}; }
inline ScopedSetting scopedDisable() { return {}; }
inline void violationIfEnabled() {}
} // namespace llvm::sys::sandbox

#endif

#endif
