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
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// MemRefMultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MemRefMultiBufferOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  SmallVector<Operation *> results;
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  for (auto *op : payloadOps) {
    bool canApplyMultiBuffer = true;
    auto target = cast<memref::AllocOp>(op);
    // Skip allocations not used in a loop.
    for (Operation *user : target->getUsers()) {
      auto loop = user->getParentOfType<LoopLikeOpInterface>();
      if (!loop) {
        canApplyMultiBuffer = false;
        break;
      }
    }
    if (!canApplyMultiBuffer)
      continue;

    auto newBuffer = memref::multiBuffer(target, getFactor());
    if (failed(newBuffer))
      return emitSilenceableFailure(target->getLoc())
             << "op failed to multibuffer";

    results.push_back(*newBuffer);
  }
  transformResults.set(getResult().cast<OpResult>(), results);
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
