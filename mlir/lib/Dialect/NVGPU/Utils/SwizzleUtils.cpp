//===- SwizzleUtils.cpp - MLIR NVGPU dialect utils for swizzling ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/NVGPU/Utils/SwizzleUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::nvgpu;

bool mlir::nvgpu::canSwizzleSharedMemoryType(MemRefType memRefType) {
  // Not support 0D MemRefs.
  if (memRefType.getRank() == 0)
    return false;

  // Only support memrefs with scalar element types (i.e., int or float).
  // Memrefs with vector element types are not supported.
  if (!memRefType.getElementType().isIntOrFloat())
    return false;

  // Check if this is necessary given the assumption of 128b accesses:
  // If dim[rank-1] is small enough to fit 8 rows in a 128B line.
  const int64_t rowSize = memRefType.getDimSize(memRefType.getRank() - 1);
  if (ShapedType::isDynamic(rowSize) || rowSize == 0)
    return false;

  const int64_t rowsPerLine =
      (8 * kSharedMemoryLineSizeBytes / memRefType.getElementTypeBitWidth()) /
      rowSize;
  const int64_t threadGroupSize =
      1LL << (7 - llvm::Log2_64(kDefaultVectorSizeBits / 8));
  if (rowsPerLine >= threadGroupSize)
    return false;

  // The XOR swizzle stays within a row only if the row size is a power of 2.
  // Otherwise it moves some accesses into the next row, where they overwrite
  // other elements.
  return llvm::isPowerOf2_64(rowSize);
}
