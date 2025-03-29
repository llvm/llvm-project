//===------- LibC.cpp - Simple implementation of libc functions --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibC.h"

#if defined(__AMDGPU__) && !defined(OMPTARGET_HAS_LIBC)
extern "C" int vprintf(const char *format, __builtin_va_list) { return -1; }
#else
extern "C" int vprintf(const char *format, __builtin_va_list);
#endif

extern "C" {
[[gnu::weak]] int memcmp(const void *lhs, const void *rhs, size_t count) {
  auto *L = reinterpret_cast<const unsigned char *>(lhs);
  auto *R = reinterpret_cast<const unsigned char *>(rhs);

  for (size_t I = 0; I < count; ++I)
    if (L[I] != R[I])
      return (int)L[I] - (int)R[I];

  return 0;
}

[[gnu::weak]] void memset(void *dst, int C, size_t count) {
  auto *dstc = reinterpret_cast<char *>(dst);
  for (size_t I = 0; I < count; ++I)
    dstc[I] = C;
}

[[gnu::weak]] int printf(const char *Format, ...) {
  __builtin_va_list vlist;
  __builtin_va_start(vlist, Format);
  return ::vprintf(Format, vlist);
}
}

namespace ompx {
[[clang::no_builtin("printf")]] int printf(const char *Format, ...) {
  __builtin_va_list vlist;
  __builtin_va_start(vlist, Format);
  return ::vprintf(Format, vlist);
}
} // namespace ompx
