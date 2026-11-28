//===- SwizzleUtils.h - MLIR NVGPU dialect utils for swizzling --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for the swizzling of shared memory accesses
// that `optimizeSharedMemoryReadsAndWrites` performs.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_NVGPU_UTILS_SWIZZLEUTILS_H
#define MLIR_DIALECT_NVGPU_UTILS_SWIZZLEUTILS_H

#include <cstdint>

namespace mlir {
class MemRefType;

namespace nvgpu {
/// The size of a shared memory line according to NV documentation.
constexpr int64_t kSharedMemoryLineSizeBytes = 128;
/// We optimize for 128bit accesses, but this can be made an argument in the
/// future.
constexpr int64_t kDefaultVectorSizeBits = 128;

/// Returns true if `optimizeSharedMemoryReadsAndWrites` can swizzle accesses to
/// shared memory buffers of type `memRefType`. This does not guarantee that the
/// memref will be swizzled, however: factors other than its type can prevent
/// it, e.g., buffers with aliases are not swizzled.
bool canSwizzleSharedMemoryType(MemRefType memRefType);
} // namespace nvgpu
} // namespace mlir

#endif // MLIR_DIALECT_NVGPU_UTILS_SWIZZLEUTILS_H
