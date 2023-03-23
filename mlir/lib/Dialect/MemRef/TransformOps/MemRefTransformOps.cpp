//===- MemRefTransformOps.cpp - Implementation of Memref transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

#define DEBUG_TYPE "memref-transforms"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

//===----------------------------------------------------------------------===//
// MemRefMultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MemRefMultiBufferOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  SmallVector<Operation *> results;
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  IRRewriter rewriter(getContext());
  for (auto *op : payloadOps) {
    bool canApplyMultiBuffer = true;
    auto target = cast<memref::AllocOp>(op);
    LLVM_DEBUG(DBGS() << "Start multibuffer transform op: " << target << "\n";);
    // Skip allocations not used in a loop.
    for (Operation *user : target->getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;
      auto loop = user->getParentOfType<LoopLikeOpInterface>();
      if (!loop) {
        LLVM_DEBUG(DBGS() << "--allocation not used in a loop\n";
                   DBGS() << "----due to user: " << *user;);
        canApplyMultiBuffer = false;
        break;
      }
    }
    if (!canApplyMultiBuffer) {
      LLVM_DEBUG(DBGS() << "--cannot apply multibuffering -> Skip\n";);
      continue;
    }

    auto newBuffer =
        memref::multiBuffer(rewriter, target, getFactor(), getSkipAnalysis());

    if (failed(newBuffer)) {
      LLVM_DEBUG(DBGS() << "--op failed to multibuffer\n";);
      return emitSilenceableFailure(target->getLoc())
             << "op failed to multibuffer";
    }

    results.push_back(*newBuffer);
  }
  transformResults.set(getResult().cast<OpResult>(), results);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MemRefExtractAddressComputationsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MemRefExtractAddressComputationsOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  memref::populateExtractAddressComputationsPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return emitDefaultDefiniteFailure(target);

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class MemRefTransformDialectExtension
    : public transform::TransformDialectExtension<
          MemRefTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();
    declareGeneratedDialect<AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<memref::MemRefDialect>();
    declareGeneratedDialect<nvgpu::NVGPUDialect>();
    declareGeneratedDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.cpp.inc"

void mlir::memref::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<MemRefTransformDialectExtension>();
}
