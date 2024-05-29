//===-------- Allocator.h - OpenMP memory allocator interface ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_ALLOCATOR_H
#define OMPTARGET_ALLOCATOR_H

#include "Types.h"

// Forward declaration.
struct KernelEnvironmentTy;

#pragma omp begin declare target device_type(nohost)

namespace ompx {

namespace allocator {

static uint64_t constexpr ALIGNMENT = 16;

/// Initialize the allocator according to \p KernelEnvironment
void init(bool IsSPMD, KernelEnvironmentTy &KernelEnvironment);

/// Allocate \p Size bytes.
[[gnu::alloc_size(1), gnu::assume_aligned(ALIGNMENT), gnu::malloc]] void *
alloc(uint64_t Size);

/// Free the allocation pointed to by \p Ptr.
void free(void *Ptr);

} // namespace allocator

} // namespace ompx

#pragma omp end declare target

#endif
