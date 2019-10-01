//===-- vfabi-demangler-fuzzer.cpp - Fuzzer VFABI using lib/Fuzzer   ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Build tool to fuzz the demangler for the vector function ABI names.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"

using namespace llvm;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  const StringRef MangledName((const char *)Data, Size);
  const auto Info = VFABI::tryDemangleForVFABI(MangledName);

  // Do not optimize away the return value. Inspired by
  // https://github.com/google/benchmark/blob/master/include/benchmark/benchmark.h#L307-L345
  asm volatile("" : : "r,m"(Info) : "memory");

  return 0;
}
