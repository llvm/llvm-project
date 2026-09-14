//===------- Offload tests - AMDGPU memory pool allocate ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMemoryPoolAllocate.h"
#include <cstdint>
#include <gtest/gtest.h>

using llvm::omp::target::plugin::classifyMemoryPoolAllocate;
using llvm::omp::target::plugin::MemoryPoolAllocateOutcome;

TEST(AMDGPUMemoryPoolAllocate, FailedStatusDoesNotInspectPointer) {
  // A failed HSA allocate may leave this undefined. Using 0x1 would also fail
  // a 16-byte alignment check, so inspecting it would be the wrong outcome.
  void *Poison = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(classifyMemoryPoolAllocate(false, &Poison, 16),
            MemoryPoolAllocateOutcome::AllocateFailed);
  EXPECT_EQ(classifyMemoryPoolAllocate(false, nullptr, 16),
            MemoryPoolAllocateOutcome::AllocateFailed);
}

TEST(AMDGPUMemoryPoolAllocate, SuccessWithAlignedPointer) {
  alignas(16) char Buffer[16];
  void *Pointer = Buffer;
  EXPECT_EQ(classifyMemoryPoolAllocate(true, &Pointer, 16),
            MemoryPoolAllocateOutcome::Success);
}

TEST(AMDGPUMemoryPoolAllocate, SuccessWithMisalignedPointer) {
  alignas(16) char Buffer[16];
  void *Pointer = Buffer + 1;
  EXPECT_EQ(classifyMemoryPoolAllocate(true, &Pointer, 16),
            MemoryPoolAllocateOutcome::PointerMisaligned);
}

TEST(AMDGPUMemoryPoolAllocate, ZeroAlignmentSkipsAlignmentCheck) {
  void *Pointer = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(classifyMemoryPoolAllocate(true, &Pointer, 0),
            MemoryPoolAllocateOutcome::Success);
}
