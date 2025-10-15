//===--- Level Zero Target RTL Implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// External and other auxilary definitions
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DEFS_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DEFS_H

#include "PluginInterface.h"
#include "Shared/Requirements.h"
#include "omptarget.h"

enum class AllocOptionTy : int32_t {
  ALLOC_OPT_NONE = 0,
  ALLOC_OPT_REDUCTION_SCRATCH = 1,
  ALLOC_OPT_REDUCTION_COUNTER = 2,
  ALLOC_OPT_HOST_MEM = 3,
  ALLOC_OPT_SLM = 4,
};

#ifndef EXTRACT_BITS
// MSB=63, LSB=0
#define EXTRACT_BITS(I64, HIGH, LOW)                                           \
  (((uint64_t)I64) >> (LOW)) & (((uint64_t)1 << ((HIGH) - (LOW) + 1)) - 1)
#endif

namespace llvm::omp::target::plugin {

/// Default alignmnet for allocation
constexpr size_t L0DefaultAlignment = 0;
/// Default staging buffer size for host to device copy (16KB)
constexpr size_t L0StagingBufferSize = (1 << 14);
/// Default staging buffer count
constexpr size_t L0StagingBufferCount = 64;
/// USM allocation threshold where preallocation does not pay off (128MB)
constexpr size_t L0UsmPreAllocThreshold = (128 << 20);
/// Host USM allocation threshold where preallocation does not pay off (8MB)
constexpr size_t L0HostUsmPreAllocThreshold = (8 << 20);

using namespace error;
/// Generic L0 handle type
using ZeHandleTy = void *;

template <typename... ArgsTy>
static Error Plugin::check(int32_t Code, const char *ErrFmt, ArgsTy... Args) {

  if (Code == OFFLOAD_SUCCESS)
    return Plugin::success();
  const char *Desc = "Unknown error";
  return createStringError<ArgsTy..., const char *>(inconvertibleErrorCode(),
                                                    ErrFmt, Args..., Desc);
}

#define L0_UNIMPLEMENTED_ERR                                                   \
  return Plugin::error(ErrorCode::UNIMPLEMENTED, "%s not implemented yet",     \
                       __func__);

} // namespace llvm::omp::target::plugin

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_LEVEL_ZERO_L0DEFS_H
