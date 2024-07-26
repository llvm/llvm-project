//===------ Sanitizer.cpp - Track allocation for sanitizer checks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "Mapping.h"
#include "Shared/Environment.h"
#include "Synchronization.h"
#include "Types.h"
#include "Utils.h"

using namespace ompx;

#define _SAN_ATTRS                                                             \
  [[clang::disable_sanitizer_instrumentation, gnu::used, gnu::retain]]
#define _SAN_ENTRY_ATTRS [[gnu::flatten, gnu::always_inline]] _SAN_ATTRS

#pragma omp begin declare target device_type(nohost)

[[gnu::visibility("protected")]] _SAN_ATTRS SanitizerEnvironmentTy
    *__sanitizer_environment_ptr;

namespace {

/// Helper to lock the sanitizer environment. While we never unlock it, this
/// allows us to have a no-op "side effect" in the spin-wait function below.
_SAN_ATTRS bool
getSanitizerEnvironmentLock(SanitizerEnvironmentTy &SE,
                            SanitizerEnvironmentTy::ErrorCodeTy ErrorCode) {
  return atomic::cas(SE.getErrorCodeLocation(), SanitizerEnvironmentTy::NONE,
                     ErrorCode, atomic::OrderingTy::seq_cst,
                     atomic::OrderingTy::seq_cst);
}

/// The spin-wait function should not be inlined, it's a catch all to give one
/// thread time to setup the sanitizer environment.
[[clang::noinline]] _SAN_ATTRS void spinWait(SanitizerEnvironmentTy &SE) {
  while (!atomic::load(&SE.IsInitialized, atomic::OrderingTy::aquire))
    ;
  __builtin_trap();
}

_SAN_ATTRS
void setLocation(SanitizerEnvironmentTy &SE, uint64_t PC) {
  for (int I = 0; I < 3; ++I) {
    SE.ThreadId[I] = mapping::getThreadIdInBlock(I);
    SE.BlockId[I] = mapping::getBlockIdInKernel(I);
  }
  SE.PC = PC;

  // This is the last step to initialize the sanitizer environment, time to
  // trap via the spinWait. Flush the memory writes and signal for the end.
  fence::system(atomic::OrderingTy::release);
  atomic::store(&SE.IsInitialized, 1, atomic::OrderingTy::release);
}

_SAN_ATTRS
void raiseExecutionError(SanitizerEnvironmentTy::ErrorCodeTy ErrorCode,
                         uint64_t PC) {
  SanitizerEnvironmentTy &SE = *__sanitizer_environment_ptr;
  bool HasLock = getSanitizerEnvironmentLock(SE, ErrorCode);

  // If no thread of this warp has the lock, end execution gracefully.
  bool AnyThreadHasLock = utils::ballotSync(lanes::All, HasLock);
  if (!AnyThreadHasLock)
    utils::terminateWarp();

  // One thread will set the location information and signal that the rest of
  // the wapr that the actual trap can be executed now.
  if (HasLock)
    setLocation(SE, PC);

  synchronize::warp(lanes::All);

  // This is not the first thread that encountered the trap, to avoid a race
  // on the sanitizer environment, this thread is simply going to spin-wait.
  // The trap above will end the program for all threads.
  spinWait(SE);
}

} // namespace

extern "C" {

_SAN_ENTRY_ATTRS void __offload_san_trap_info(uint64_t PC) {
  raiseExecutionError(SanitizerEnvironmentTy::TRAP, PC);
}

_SAN_ENTRY_ATTRS void __offload_san_unreachable_info(uint64_t PC) {
  raiseExecutionError(SanitizerEnvironmentTy::UNREACHABLE, PC);
}
}

#pragma omp end declare target
