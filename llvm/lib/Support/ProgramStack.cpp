//===--- RunOnNewStack.cpp - Crash Recovery -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ProgramStack.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"

#ifdef LLVM_ON_UNIX
# include <sys/resource.h> // for getrlimit
#endif

#ifdef _MSC_VER
# include <intrin.h>  // for _AddressOfReturnAddress
#endif

#ifndef LLVM_HAS_SPLIT_STACKS
# include "llvm/Support/thread.h"
#endif

using namespace llvm;

uintptr_t llvm::getStackPointer() {
#if __GNUC__ || __has_builtin(__builtin_frame_address)
  return (uintptr_t)__builtin_frame_address(0);
#elif defined(_MSC_VER)
  return (uintptr_t)_AddressOfReturnAddress();
#else
  volatile char CharOnStack = 0;
  // The volatile store here is intended to escape the local variable, to
  // prevent the compiler from optimizing CharOnStack into anything other
  // than a char on the stack.
  //
  // Tested on: MSVC 2015 - 2019, GCC 4.9 - 9, Clang 3.2 - 9, ICC 13 - 19.
  char *volatile Ptr = &CharOnStack;
  return (uintptr_t)Ptr;
#endif
}

unsigned llvm::getDefaultStackSize() {
#ifdef LLVM_ON_UNIX
  rlimit RL;
  getrlimit(RLIMIT_STACK, &RL);
  return RL.rlim_cur;
#else
  // Clang recursively parses, instantiates templates, and evaluates constant
  // expressions. We've found 8MiB to be a reasonable stack size given the way
  // Clang works and the way C++ is commonly written.
  return 8 << 20;
#endif
}

// Not an anonymous namespace to avoid warning about undefined local function.
namespace llvm {
#ifdef LLVM_HAS_SPLIT_STACKS_AARCH64
void runOnNewStackImpl(void *Stack, void (*Fn)(void *), void *Ctx) __asm__(
    "_ZN4llvm17runOnNewStackImplEPvPFvS0_ES0_");

// This can't use naked functions because there is no way to know if cfi
// directives are being emitted or not.
//
// When adding new platforms it may be better to move to a .S file with macros
// for dealing with platform differences.
__asm__ (
    ".globl  _ZN4llvm17runOnNewStackImplEPvPFvS0_ES0_\n\t"
    ".p2align  2\n\t"
    "_ZN4llvm17runOnNewStackImplEPvPFvS0_ES0_:\n\t"
    ".cfi_startproc\n\t"
    "mov       x16, sp\n\t"
    "sub       x0, x0, #0x20\n\t"            // subtract space from stack
    "stp       xzr, x16, [x0, #0x00]\n\t"    // save old sp
    "stp       x29, x30, [x0, #0x10]\n\t"    // save fp, lr
    "mov       sp, x0\n\t"                   // switch to new stack
    "add       x29, x0, #0x10\n\t"           // switch to new frame
    ".cfi_def_cfa w29, 16\n\t"
    ".cfi_offset w30, -8\n\t"                // lr
    ".cfi_offset w29, -16\n\t"               // fp

    "mov       x0, x2\n\t"                   // Ctx is the only argument
    "blr       x1\n\t"                       // call Fn

    "ldp       x29, x30, [sp, #0x10]\n\t"    // restore fp, lr
    "ldp       xzr, x16, [sp, #0x00]\n\t"    // load old sp
    "mov       sp, x16\n\t"
    "ret\n\t"
    ".cfi_endproc"
);
#endif
} // namespace llvm

namespace {
#ifdef LLVM_HAS_SPLIT_STACKS
void callback(void *Ctx) {
  (*reinterpret_cast<function_ref<void()> *>(Ctx))();
}
#endif
} // namespace

#ifdef LLVM_HAS_SPLIT_STACKS
void llvm::runOnNewStack(unsigned StackSize, function_ref<void()> Fn) {
  if (StackSize == 0)
    StackSize = getDefaultStackSize();

  // We use malloc here instead of mmap because:
  //   - it's simpler,
  //   - many malloc implementations will reuse the allocation in cases where
  //     we're bouncing accross the edge of a stack boundry, and
  //   - many malloc implemenations will already provide guard pages for
  //     allocations this large.
  void *Stack = malloc(StackSize);
  void *BottomOfStack = (char *)Stack + StackSize;

  runOnNewStackImpl(BottomOfStack, callback, &Fn);

  free(Stack);
}
#else
void llvm::runOnNewStack(unsigned StackSize, function_ref<void()> Fn) {
  llvm::thread Thread(
      StackSize == 0 ? std::nullopt : std::optional<unsigned>(StackSize), Fn);
  Thread.join();
}
#endif
