//===--- ProgramStack.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PROGRAMSTACK_H
#define LLVM_SUPPORT_PROGRAMSTACK_H

#include "llvm/ADT/STLFunctionalExtras.h"

// LLVM_HAS_SPLIT_STACKS is exposed in the header because CrashRecoveryContext
// needs to know if it's running on another thread or not.
//
// Currently only Apple AArch64 is known to support split stacks in the debugger
// and other tooling.
#if defined(__APPLE__) && defined(__MACH__) && defined(__aarch64__) &&         \
    __has_extension(gnu_asm)
# define LLVM_HAS_SPLIT_STACKS
# define LLVM_HAS_SPLIT_STACKS_AARCH64
#endif

namespace llvm {

/// \returns an address close to the current value of the stack pointer.
///
/// The value is not guaranteed to point to anything specific. It can be used to
/// estimate how much stack space has been used since the previous call.
uintptr_t getStackPointer();

/// \returns the default stack size for this platform.
///
/// Based on \p RLIMIT_STACK or the equivalent.
unsigned getDefaultStackSize();

/// Runs Fn on a new stack of at least the given size.
///
/// \param StackSize requested stack size. A size of 0 uses the default stack
///                  size of the platform.
///
/// The preferred implementation is split stacks on platforms that have a good
/// debugging experience for them. On other platforms a new thread is used.
void runOnNewStack(unsigned StackSize, function_ref<void()> Fn);

template <typename R, typename... Ts>
std::enable_if_t<!std::is_same_v<R, void>, R>
runOnNewStack(unsigned StackSize, function_ref<R(Ts...)> Fn, Ts &&...Args) {
  std::optional<R> Ret;
  runOnNewStack(StackSize, [&]() { Ret = Fn(std::forward<Ts>(Args)...); });
  return std::move(*Ret);
}

template <typename... Ts>
void runOnNewStack(unsigned StackSize, function_ref<void(Ts...)> Fn,
                   Ts &&...Args) {
  runOnNewStack(StackSize, [&]() { Fn(std::forward<Ts>(Args)...); });
}

} // namespace llvm

#endif // LLVM_SUPPORT_PROGRAMSTACK_H
