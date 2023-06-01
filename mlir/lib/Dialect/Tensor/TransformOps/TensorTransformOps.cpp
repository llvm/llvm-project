//===- TensorTransformOps.cpp - Implementation of tensor transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace tensor;

//===----------------------------------------------------------------------===//
// TrackingListener
//===----------------------------------------------------------------------===//

Operation *
tensor::TrackingListener::findReplacementOp(Operation *op,
                                            ValueRange newValues) const {
  SmallVector<Value> values(newValues.begin(), newValues.end());
  do {
    if (Operation *replacement =
            transform::TrackingListener::findReplacementOp(op, values))
      return replacement;

    Operation *defOp = getCommonDefiningOp(values);
    if (!defOp)
      return nullptr;

    // Skip cast-like operations.
    values.clear();
    llvm::TypeSwitch<Operation *>(defOp)
        .Case<CastOp>([&](CastOp op) { values.push_back(op.getSource()); })
        .Case<CollapseShapeOp>(
            [&](CollapseShapeOp op) { values.push_back(op.getSrc()); })
        .Case<ExpandShapeOp>(
            [&](ExpandShapeOp op) { values.push_back(op.getSrc()); })
        .Case<ReshapeOp>(
            [&](ReshapeOp op) { values.push_back(op.getSource()); })
        .Case<InsertSliceOp>([&](InsertSliceOp op) {
          if (isCastLikeInsertSliceOp(op))
            values.push_back(op.getSource());
        })
        .Case<ExtractSliceOp>([&](ExtractSliceOp op) {
          if (isCastLikeExtractSliceOp(op))
            values.push_back(op.getSource());
        })
        .Default([](Operation *op) {});
  } while (!values.empty());

  return nullptr;
}

//===----------------------------------------------------------------------===//
// MakeLoopIndependentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MakeLoopIndependentOp::applyToOne(
    Operation *target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Gather IVs.
  SmallVector<Value> ivs;
  Operation *nextOp = target;
  for (uint64_t i = 0, e = getNumLoops(); i < e; ++i) {
    nextOp = nextOp->getParentOfType<scf::ForOp>();
    if (!nextOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "could not find " << i
                                         << "-th enclosing loop";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    ivs.push_back(cast<scf::ForOp>(nextOp).getInductionVar());
  }

  // Rewrite IR.
  IRRewriter rewriter(target->getContext());
  FailureOr<Value> replacement = failure();
  if (auto padOp = dyn_cast<tensor::PadOp>(target)) {
    replacement = tensor::buildIndependentOp(rewriter, padOp, ivs);
  } else if (auto emptyOp = dyn_cast<tensor::EmptyOp>(target)) {
    replacement = tensor::buildIndependentOp(rewriter, emptyOp, ivs);
  } else {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "unsupported target op";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  if (failed(replacement)) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError() << "could not make target op loop-independent";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  rewriter.replaceOp(target, *replacement);
  results.push_back(replacement->getDefiningOp());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class TensorTransformDialectExtension
    : public transform::TransformDialectExtension<
          TensorTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.cpp.inc"

void mlir::tensor::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<TensorTransformDialectExtension>();
}
