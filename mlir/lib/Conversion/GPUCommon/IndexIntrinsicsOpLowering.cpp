//===- IndexIntrinsicsOpLowering.cpp - GPU Index Op Lowering --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IndexIntrinsicsOpLowering.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::gpu::index_lowering;

LLVM::ConstantRangeAttr mlir::gpu::index_lowering::getIndexOpRange(
    Operation *op, gpu::Dimension dim, std::optional<uint32_t> opUpperBound,
    IndexKind indexKind, IntrType intrType, unsigned bitWidth) {
  // Order of priority for bounds:
  // 1. The upper_bound attribute
  // 2. Inherent attributes on a surrounding gpu.func
  // 3. Discardable attributes on a surrounding function of any kind
  // The below code handles these in reverse order so that more important
  // sources overwrite less important ones. As an exception, dimension-size
  // getters will return exact bounds if known.
  std::optional<uint32_t> upperBound =
      getKnownDimensionSizeAround(op, indexKind, dim);
  // If our upper bound is the maximum possible value, we can't easily construct
  // the constant range for it.
  if (upperBound && intrType == IntrType::Dim &&
      *upperBound < std::numeric_limits<uint32_t>::max())
    return LLVM::ConstantRangeAttr::get(op->getContext(), bitWidth, *upperBound,
                                        *upperBound + 1);

  if (opUpperBound)
    upperBound = *opUpperBound;

  if (!upperBound || intrType == IntrType::None)
    return nullptr;

  uint32_t min = (intrType == IntrType::Dim ? 1u : 0u);
  uint32_t max =
      llvm::SaturatingAdd(*upperBound, (intrType == IntrType::Id ? 0u : 1u));
  return LLVM::ConstantRangeAttr::get(op->getContext(), bitWidth, min, max);
}
