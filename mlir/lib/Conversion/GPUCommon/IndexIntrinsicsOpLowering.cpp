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
  // sources overwrite less important ones.
  DenseI32ArrayAttr funcBounds = nullptr;
  if (auto funcOp = op->getParentOfType<FunctionOpInterface>()) {
    switch (indexKind) {
    case IndexKind::Block: {
      auto blockHelper =
          gpu::GPUDialect::KnownBlockSizeAttrHelper(op->getContext());
      if (blockHelper.isAttrPresent(funcOp))
        funcBounds = blockHelper.getAttr(funcOp);
      break;
    }
    case IndexKind::Grid: {
      auto gridHelper =
          gpu::GPUDialect::KnownGridSizeAttrHelper(op->getContext());
      if (gridHelper.isAttrPresent(funcOp))
        funcBounds = gridHelper.getAttr(funcOp);
      break;
    }
    case IndexKind::Cluster: {
      auto clusterHelper =
          gpu::GPUDialect::KnownClusterSizeAttrHelper(op->getContext());
      if (clusterHelper.isAttrPresent(funcOp))
        funcBounds = clusterHelper.getAttr(funcOp);
      break;
    }
    case IndexKind::Other:
      break;
    }
  }
  if (auto gpuFunc = op->getParentOfType<gpu::GPUFuncOp>()) {
    switch (indexKind) {
    case IndexKind::Block:
      funcBounds = gpuFunc.getKnownBlockSizeAttr();
      break;
    case IndexKind::Grid:
      funcBounds = gpuFunc.getKnownGridSizeAttr();
      break;
    case IndexKind::Cluster:
      funcBounds = gpuFunc.getKnownClusterSizeAttr();
      break;
    case IndexKind::Other:
      break;
    }
  }
  std::optional<uint32_t> upperBound;
  if (funcBounds)
    upperBound = funcBounds.asArrayRef()[static_cast<uint32_t>(dim)];
  if (opUpperBound)
    upperBound = *opUpperBound;

  if (!upperBound || intrType == IntrType::None)
    return nullptr;

  uint32_t min = (intrType == IntrType::Dim ? 1u : 0u);
  uint32_t max =
      llvm::SaturatingAdd(*upperBound, (intrType == IntrType::Id ? 0u : 1u));
  return LLVM::ConstantRangeAttr::get(op->getContext(), bitWidth, min, max);
}
