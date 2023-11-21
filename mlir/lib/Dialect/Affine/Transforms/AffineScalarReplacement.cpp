//===- AffineScalarReplacement.cpp - Affine scalar replacement pass -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward affine memref stores to loads, thereby
// potentially getting rid of intermediate memrefs entirely. It also removes
// redundant loads.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LogicalResult.h"
#include <algorithm>

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINESCALARREPLACEMENT
#include "mlir/Dialect/Affine/Passes.h.inc"
} // namespace affine
} // namespace mlir

#define DEBUG_TYPE "affine-scalrep"

using namespace mlir;
using namespace mlir::affine;

namespace {
struct AffineScalarReplacement
    : public affine::impl::AffineScalarReplacementBase<
          AffineScalarReplacement> {
  void runOnOperation() override;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::affine::createAffineScalarReplacementPass() {
  return std::make_unique<AffineScalarReplacement>();
}

void AffineScalarReplacement::runOnOperation() {
  affineScalarReplace(getOperation(), getAnalysis<DominanceInfo>(),
                      getAnalysis<PostDominanceInfo>());
}
