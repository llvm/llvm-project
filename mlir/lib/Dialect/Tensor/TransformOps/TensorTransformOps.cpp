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
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace tensor;

//===----------------------------------------------------------------------===//
// FindPayloadReplacementOpInterface implementations
//===----------------------------------------------------------------------===//

namespace {
struct ExtractSliceOpReplacementInterface
    : public transform::FindPayloadReplacementOpInterface::ExternalModel<
          ExtractSliceOpReplacementInterface, tensor::ExtractSliceOp> {
  SmallVector<Value> getNextOperands(Operation *op) const {
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    if (!isCastLikeExtractSliceOp(extractSliceOp))
      return {};
    return {extractSliceOp.getSource()};
  }
};

struct InsertSliceOpReplacementInterface
    : public transform::FindPayloadReplacementOpInterface::ExternalModel<
          InsertSliceOpReplacementInterface, tensor::InsertSliceOp> {
  SmallVector<Value> getNextOperands(Operation *op) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    if (!isCastLikeInsertSliceOp(insertSliceOp))
      return {};
    return {insertSliceOp.getSource()};
  }
};

struct ReshapeOpReplacementInterface
    : public transform::FindPayloadReplacementOpInterface::ExternalModel<
          ReshapeOpReplacementInterface, tensor::ReshapeOp> {
  SmallVector<Value> getNextOperands(Operation *op) const {
    auto reshapeOp = cast<tensor::ReshapeOp>(op);
    return {reshapeOp.getSource()};
  }
};

template <typename ConcreteOp>
struct ReassociativeReshapeOpReplacementInterface
    : public transform::FindPayloadReplacementOpInterface::ExternalModel<
          ReassociativeReshapeOpReplacementInterface<ConcreteOp>, ConcreteOp> {
  SmallVector<Value> getNextOperands(Operation *op) const {
    auto reshapeOp = cast<ConcreteOp>(op);
    return {reshapeOp.getSrc()};
  }
};
} // namespace

void tensor::registerFindPayloadReplacementOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    CollapseShapeOp::attachInterface<
        ReassociativeReshapeOpReplacementInterface<CollapseShapeOp>>(*ctx);
    ExpandShapeOp::attachInterface<
        ReassociativeReshapeOpReplacementInterface<ExpandShapeOp>>(*ctx);
    ExtractSliceOp::attachInterface<ExtractSliceOpReplacementInterface>(*ctx);
    InsertSliceOp::attachInterface<InsertSliceOpReplacementInterface>(*ctx);
    ReshapeOp::attachInterface<ReshapeOpReplacementInterface>(*ctx);
  });
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyDecomposeTensorConcatPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  tensor::populateDecomposeTensorConcatPatterns(patterns);
}

void transform::ApplyDropRedundantInsertSliceRankExpansionPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  tensor::populateDropRedundantInsertSliceRankExpansionPatterns(patterns);
}

void transform::ApplyFoldTensorEmptyPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  tensor::populateFoldTensorEmptyPatterns(patterns, getFoldSingleUseOnly());
}

void transform::ApplyFoldIntoPackAndUnpackPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  tensor::populateFoldIntoPackAndUnpackPatterns(patterns);
}

void transform::ApplyFoldTensorSubsetOpsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  tensor::populateFoldTensorSubsetOpPatterns(patterns);
}

void transform::ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
}

void transform::ApplyMergeConsecutiveInsertExtractSlicePatternsOp::
    populatePatterns(RewritePatternSet &patterns) {
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
}

void transform::ApplyReassociativeReshapeFoldingPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}

void transform::ApplyRewriteTensorOpsAsConstantPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  tensor::populateRewriteAsConstantPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// TypeConversionCastTensorShapeOp
//===----------------------------------------------------------------------===//

void transform::TypeConversionCastShapeDynamicDimsOp::
    populateTypeMaterializations(TypeConverter &converter) {
  bool ignoreDynamicInfo = getIgnoreDynamicInfo();
  converter.addSourceMaterialization([ignoreDynamicInfo](
                                         OpBuilder &builder, Type resultType,
                                         ValueRange inputs,
                                         Location loc) -> std::optional<Value> {
    if (inputs.size() != 1) {
      return std::nullopt;
    }
    Value input = inputs[0];
    if (!ignoreDynamicInfo &&
        !tensor::preservesStaticInformation(resultType, input.getType())) {
      return std::nullopt;
    }
    if (!tensor::CastOp::areCastCompatible(input.getType(), resultType)) {
      return std::nullopt;
    }
    return builder.create<tensor::CastOp>(loc, resultType, input).getResult();
  });
  converter.addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                        ValueRange inputs,
                                        Location loc) -> std::optional<Value> {
    if (inputs.size() != 1) {
      return std::nullopt;
    }
    Value input = inputs[0];
    if (!tensor::CastOp::areCastCompatible(input.getType(), resultType)) {
      return std::nullopt;
    }
    return builder.create<tensor::CastOp>(loc, resultType, input).getResult();
  });
}

//===----------------------------------------------------------------------===//
// MakeLoopIndependentOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MakeLoopIndependentOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
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
