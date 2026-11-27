//===- TosaPlanarBlockScaled.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Conversions to and from the TOSA dialect may require different
// representations of block-scaled tensors. Operations such as
// linalg.scaled_contraction expect data and scales as separate tensor inputs,
// referred to here as the planar format. In contrast, the composite format
// represents data and scales in a single block-scaled tensor as expected by the
// TOSA specification.
//
// This file implements the pass for materializing the planar format:
// - tosa-materialize-planar-block-scaled exposes the data and scale components
//   of composite block-scaled tensors as separate SSA values by inserting
//   tosa.decompose_block_scaled and tosa.compose_block_scaled operations.
//   This provides a planar interface for subsequent lowering to backends such
//   as linalg that expect separate data and scale tensors.
//
// Known Limitations:
// - Multiple functions and function calls are not currently supported.
// - Functions with multiple returns are not currently supported.
// - Constant dense resources are not currently supported.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAMATERIALIZEPLANARBLOCKSCALEDPASS
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"

namespace {
struct PlanarTypePair {
  TensorType dataType;
  TensorType scaleType;
};

static FailureOr<PlanarTypePair> getPlanarTypes(Type type) {
  const auto tensorType = dyn_cast<TensorType>(type);
  if (!tensorType)
    return failure();

  const auto blockScaledType =
      dyn_cast<BlockScaledType>(tensorType.getElementType());
  if (!blockScaledType)
    return failure();

  if (const auto rankedType = dyn_cast<RankedTensorType>(tensorType)) {
    SmallVector<int64_t> newScaleShape(rankedType.getShape());
    int64_t innermostDim = newScaleShape.back();
    if (!ShapedType::isDynamic(innermostDim))
      innermostDim = llvm::divideCeil(
          innermostDim, static_cast<int64_t>(BlockShapeAttr::getBlockShapeValue(
                            blockScaledType.getBlockShape())));
    newScaleShape.back() = innermostDim;

    return PlanarTypePair{RankedTensorType::get(rankedType.getShape(),
                                                blockScaledType.getValueType(),
                                                rankedType.getEncoding()),
                          RankedTensorType::get(newScaleShape,
                                                blockScaledType.getScaleType(),
                                                rankedType.getEncoding())};
  }

  return PlanarTypePair{
      UnrankedTensorType::get(blockScaledType.getValueType()),
      UnrankedTensorType::get(blockScaledType.getScaleType())};
}

static ComposeBlockScaledOp createCompose(OpBuilder &builder, Location loc,
                                          Type packedType, Value data,
                                          Value scale) {
  const auto tensorType = cast<TensorType>(packedType);
  const auto blockScaledType =
      cast<BlockScaledType>(tensorType.getElementType());
  return ComposeBlockScaledOp::create(
      builder, loc, packedType, data, scale,
      BlockShapeAttr::get(builder.getContext(),
                          blockScaledType.getBlockShape()));
}

static DecomposeBlockScaledOp createDecompose(OpBuilder &builder, Location loc,
                                              Value packed) {
  const FailureOr<PlanarTypePair> planarTypes =
      getPlanarTypes(packed.getType());
  return DecomposeBlockScaledOp::create(builder, loc, planarTypes->dataType,
                                        planarTypes->scaleType, packed);
}

static bool hasPlanarSurroundingAPI(Operation *op) {
  for (Value operand : op->getOperands()) {
    const auto tensorType = dyn_cast<TensorType>(operand.getType());
    if (!tensorType || !isa<BlockScaledType>(tensorType.getElementType()))
      continue;
    if (!operand.getDefiningOp<ComposeBlockScaledOp>())
      return false;
  }

  for (Value result : op->getResults()) {
    const auto tensorType = dyn_cast<TensorType>(result.getType());
    if (!tensorType || !isa<BlockScaledType>(tensorType.getElementType()))
      continue;
    if (!llvm::all_of(result.getUsers(), [](Operation *user) {
          return isa<DecomposeBlockScaledOp>(user);
        }))
      return false;
  }
  return true;
}

class ConvertTosaOpToPlanarAPI final : public ConversionPattern {
public:
  ConvertTosaOpToPlanarAPI(const TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag{}, 1, context) {
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(Operation *op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> sourceOperands;
    sourceOperands.reserve(op->getNumOperands());
    IRMapping mapping;
    for (auto [original, converted] :
         llvm::zip_equal(op->getOperands(), adaptor)) {
      Value replacement = typeConverter->materializeSourceConversion(
          rewriter, op->getLoc(), original.getType(), converted);
      mapping.map(original, replacement);
    }
    Operation *newOp = rewriter.clone(*op, mapping);

    SmallVector<SmallVector<Value>> replacements;
    replacements.reserve(op->getNumResults());
    for (auto [original, replacement] :
         llvm::zip_equal(op->getResults(), newOp->getResults())) {
      SmallVector<Type> convertedTypes;
      if (failed(
              typeConverter->convertTypes(original.getType(), convertedTypes)))
        return failure();
      SmallVector<Value> convertedValues =
          typeConverter->materializeTargetConversion(
              rewriter, op->getLoc(), convertedTypes, replacement,
              original.getType());
      if (convertedValues.size() != convertedTypes.size())
        return failure();
      replacements.push_back(std::move(convertedValues));
    }

    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};
} // namespace

class TosaMaterializePlanarBlockScaledPass
    : public impl::TosaMaterializePlanarBlockScaledPassBase<
          TosaMaterializePlanarBlockScaledPass> {

public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion(
        [](TensorType type,
           SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          if (!isa<BlockScaledType>(type.getElementType()))
            return std::nullopt;

          const FailureOr<PlanarTypePair> planarTypes = getPlanarTypes(type);
          if (failed(planarTypes))
            return failure();
          results.push_back(planarTypes->dataType);
          results.push_back(planarTypes->scaleType);
          return success();
        });

    converter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      if (inputs.size() == 1 && inputs.front().getType() == resultType)
        return inputs.front();

      const FailureOr<PlanarTypePair> planarTypes = getPlanarTypes(resultType);
      if (failed(planarTypes) || inputs.size() != 2 ||
          inputs[0].getType() != planarTypes->dataType ||
          inputs[1].getType() != planarTypes->scaleType)
        return {};
      return createCompose(builder, loc, resultType, inputs[0], inputs[1]);
    });
    converter.addTargetMaterialization(
        [](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
           Location loc) -> SmallVector<Value> {
          if (inputs.size() != 1)
            return {};
          if (resultTypes.size() == 1 &&
              inputs.front().getType() == resultTypes.front())
            return {inputs.front()};

          const auto inputType = dyn_cast<TensorType>(inputs.front().getType());
          if (!inputType || !isa<BlockScaledType>(inputType.getElementType()))
            return {};

          const FailureOr<PlanarTypePair> planarTypes =
              getPlanarTypes(inputs.front().getType());
          if (failed(planarTypes) || resultTypes.size() != 2 ||
              resultTypes[0] != planarTypes->dataType ||
              resultTypes[1] != planarTypes->scaleType)
            return {};

          DecomposeBlockScaledOp decompose =
              createDecompose(builder, loc, inputs.front());
          return {decompose.getData(), decompose.getScale()};
        });

    ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalDialect<TosaDialect>([](Operation *op) {
      return isa<ComposeBlockScaledOp, DecomposeBlockScaledOp>(op) ||
             hasPlanarSurroundingAPI(op);
    });
    if (convertFunctionBoundaries) {
      target.addDynamicallyLegalOp<func::FuncOp>([&converter](func::FuncOp op) {
        return converter.isSignatureLegal(op.getFunctionType()) &&
               converter.isLegal(&op.getBody());
      });
      target.addDynamicallyLegalOp<func::CallOp>(
          [&converter](func::CallOp op) { return converter.isLegal(op); });
      target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
        const FunctionType functionType =
            op->getParentOfType<func::FuncOp>().getFunctionType();
        return llvm::equal(op.getOperandTypes(), functionType.getResults());
      });
    }

    RewritePatternSet patterns(context);
    if (convertFunctionBoundaries) {
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                     converter);
      populateCallOpTypeConversionPattern(patterns, converter);
      populateReturnOpTypeConversionPattern(patterns, converter);
    }
    patterns.add<ConvertTosaOpToPlanarAPI>(converter, context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace tosa
} // namespace mlir
