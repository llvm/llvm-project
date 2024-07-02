//===------- LibC.cpp - Simple implementation of libc functions --- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibC.h"

#pragma omp begin declare target device_type(nohost)

namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t);
}

#pragma omp begin declare variant match(                                       \
        device = {arch(nvptx, nvptx64)},                                       \
            implementation = {extension(match_any)})
extern "C" int32_t vprintf(const char *, void *);
namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return vprintf(Format, Arguments);
}
} // namespace impl
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {arch(amdgcn)})

#ifdef OMPTARGET_HAS_LIBC
// TODO: Remove this handling once we have varargs support.
extern "C" struct FILE *stdout;
extern "C" int32_t rpc_fprintf(FILE *, const char *, void *, uint64_t);

namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t Size) {
  return rpc_fprintf(stdout, Format, Arguments, Size);
}
} // namespace impl
#else
// We do not have a vprintf implementation for AMD GPU so we use a stub.
namespace impl {
int32_t omp_vprintf(const char *Format, void *Arguments, uint32_t) {
  return -1;
}
} // namespace impl
#endif
#pragma omp end declare variant

extern "C" {

int memcmp(const void *lhs, const void *rhs, size_t count) {
  auto *L = reinterpret_cast<const unsigned char *>(lhs);
  auto *R = reinterpret_cast<const unsigned char *>(rhs);

  for (size_t I = 0; I < count; ++I)
    if (L[I] != R[I])
      return (int)L[I] - (int)R[I];

  return 0;
}

void memset(void *dst, int C, size_t count) {
  auto *dstc = reinterpret_cast<char *>(dst);
  for (size_t I = 0; I < count; ++I)
    dstc[I] = C;
}

/// printf() calls are rewritten by CGGPUBuiltin to __llvm_omp_vprintf
int32_t __llvm_omp_vprintf(const char *Format, void *Arguments, uint32_t Size) {
  return impl::omp_vprintf(Format, Arguments, Size);
}
}

#pragma omp end declare target
