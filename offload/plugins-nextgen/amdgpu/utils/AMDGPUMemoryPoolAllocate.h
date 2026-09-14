//===- AMDGPUMemoryPoolAllocate.h - HSA pool allocate result ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_UTILS_AMDGPUMEMORYPOOLALLOCATE_H
#define OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_UTILS_AMDGPUMEMORYPOOLALLOCATE_H

#include "llvm/Support/Alignment.h"
#include <cstddef>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Result of an HSA memory-pool allocate after the runtime call returns.
enum class MemoryPoolAllocateOutcome {
  AllocateFailed,
  PointerMisaligned,
  Success,
};

/// Classify a pool allocate. \p PointerStorage is read only when
/// \p AllocateSucceeded is true, because a failed HSA allocate may leave the
/// output pointer undefined.
inline MemoryPoolAllocateOutcome
classifyMemoryPoolAllocate(bool AllocateSucceeded, void *const *PointerStorage,
                           size_t Alignment) {
  if (!AllocateSucceeded)
    return MemoryPoolAllocateOutcome::AllocateFailed;

  if (Alignment > 0 &&
      !llvm::isAddrAligned(llvm::Align(Alignment), *PointerStorage))
    return MemoryPoolAllocateOutcome::PointerMisaligned;

  return MemoryPoolAllocateOutcome::Success;
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_UTILS_AMDGPUMEMORYPOOLALLOCATE_H
