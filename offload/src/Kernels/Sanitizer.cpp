//===-- Kenrels/Sanitizer.cpp - Sanitizer Kernel Definitions --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include <cstdint>

extern "C" {
__device__ void __sanitizer_register_host(void *P, uint64_t Bytes,
                                          uint64_t Slot);
__device__ void __sanitizer_unregister_host(void *P);

[[clang::disable_sanitizer_instrumentation]] __global__ void
__sanitizer_register(void *P, uint64_t Bytes, uint64_t Slot) {
  __sanitizer_register_host(P, Bytes, Slot);
}

[[clang::disable_sanitizer_instrumentation]] __global__ void
__sanitizer_unregister(void *P) {
  __sanitizer_unregister_host(P);
}
}
