//===- AffineParallelize.cpp - Affineparallelize Pass---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parallelizer for affine loop nests that is able to
// perform inner or outer loop parallelization.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Passes.h.inc"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-parallel"

using namespace mlir;

namespace {
/// Convert all parallel affine.for op into 1-D affine.parallel op.
struct AffineParallelize : public AffineParallelizeBase<AffineParallelize> {
  void runOnFunction() override;
};
} // namespace

void AffineParallelize::runOnFunction() {
  FuncOp f = getFunction();
  SmallVector<AffineForOp, 8> parallelizableLoops;
  f.walk([&](AffineForOp loop) {
    if (isLoopParallel(loop))
      parallelizableLoops.push_back(loop);
  });
  for (AffineForOp loop : parallelizableLoops)
    affineParallelize(loop);
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createAffineParallelizePass() {
  return std::make_unique<AffineParallelize>();
}
