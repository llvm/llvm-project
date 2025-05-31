//===- LoopExtensionOps.cpp - Loop extension for the Transform dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.cpp.inc"

//===----------------------------------------------------------------------===//
// HoistLoopInvariantSubsetsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::HoistLoopInvariantSubsetsOp::applyToOne(
    transform::TransformRewriter &rewriter, LoopLikeOpInterface loopLikeOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  hoistLoopInvariantSubsets(rewriter, loopLikeOp);
  return DiagnosedSilenceableFailure::success();
}

void transform::HoistLoopInvariantSubsetsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::modifiesPayload(effects);
}
