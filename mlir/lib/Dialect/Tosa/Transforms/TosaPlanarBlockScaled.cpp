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
// This file implements two complementary passes for converting between these
// formats:
// - tosa-materialize-planar-block-scaled exposes the data and scale components
// of
//   composite block-scaled tensors as separate SSA values by inserting
//   tosa.decompose_block_scaled and tosa.compose_block_scaled operations.
//   This provides a planar interface for subsequent lowering to backends such
//   as linalg that expect separate data and scale tensors.
// - tosa-remove-planar-block-scaled removes tosa.decompose_block_scaled and
//   tosa.compose_block_scaled operations to convert from the planar format to
//   the composite format. This is useful when legalizing operations to the TOSA
//   dialect.
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAMATERIALIZEPLANARBLOCKSCALEDPASS
#define GEN_PASS_DEF_TOSAREMOVEPLANARBLOCKSCALEDPASS
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

struct ConvertConstToPlanar : public OpConversionPattern<ConstOp> {
  ConvertConstToPlanar(const TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern(converter, context, /*benefit=*/2) {}

  LogicalResult
  matchAndRewrite(ConstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto values = dyn_cast<DenseElementsAttr>(op.getValues());
    if (!values)
      return rewriter.notifyMatchFailure(op, "expected dense constant values");
    const auto blockScaledType =
        dyn_cast<BlockScaledType>(values.getElementType());
    if (!blockScaledType || !blockScaledType.hasScaleValues())
      return rewriter.notifyMatchFailure(
          op, "expected block-scaled constant values with embedded scales");

    const FailureOr<PlanarTypePair> planarTypes = getPlanarTypes(op.getType());
    if (failed(planarTypes))
      return rewriter.notifyMatchFailure(
          op, "could not derive planar types from the constant result type");

    const DenseElementsAttr dataValues =
        values.bitcast(blockScaledType.getValueType());
    const auto scaleValues = DenseElementsAttr::get(
        planarTypes->scaleType, blockScaledType.getScaleValues());
    auto data = ConstOp::create(rewriter, op.getLoc(), planarTypes->dataType,
                                dataValues);
    auto scale = ConstOp::create(rewriter, op.getLoc(), planarTypes->scaleType,
                                 scaleValues);
    rewriter.replaceOpWithMultiple(op, {{data.getResult(), scale.getResult()}});
    return success();
  }
};

struct ComposeBlockScaledConstants
    : public OpRewritePattern<ComposeBlockScaledOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ComposeBlockScaledOp op,
                                PatternRewriter &rewriter) const override {
    auto data = op.getData().getDefiningOp<ConstOp>();
    auto scale = op.getScale().getDefiningOp<ConstOp>();
    if (!data || !scale)
      return rewriter.notifyMatchFailure(
          op, "expected data and scale operands defined by tosa.const");
    auto dataValues = dyn_cast<DenseElementsAttr>(data.getValues());
    auto scaleValues = dyn_cast<DenseElementsAttr>(scale.getValues());
    if (!dataValues || !scaleValues)
      return rewriter.notifyMatchFailure(
          op, "expected dense data and scale constant values");

    const auto blockScaledType =
        cast<BlockScaledType>(cast<TensorType>(op.getType()).getElementType());

    const ShapedType attrType = dataValues.getType().clone(blockScaledType);
    if (failed(verifyBlockScaledTensorType(attrType, {})))
      return rewriter.notifyMatchFailure(
          op, "data constant shape is invalid for the block-scaled type");
    const FailureOr<PlanarTypePair> planarTypes = getPlanarTypes(attrType);
    if (failed(planarTypes))
      return rewriter.notifyMatchFailure(
          op, "could not derive planar types from the compound attribute type");

    SmallVector<Attribute> scales(scaleValues.getValues<Attribute>());
    const BlockScaledType constantType = BlockScaledType::get(
        rewriter.getContext(), blockScaledType.getBlockShape(),
        blockScaledType.getScaleType(), blockScaledType.getValueType(), scales);
    rewriter.replaceOpWithNewOp<ConstOp>(op, op.getType(),
                                         dataValues.bitcast(constantType));
    return success();
  }
};

struct RemoveDecomposeComposeBlockScaled
    : public OpRewritePattern<ComposeBlockScaledOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ComposeBlockScaledOp op,
                                PatternRewriter &rewriter) const override {
    auto decompose = op.getData().getDefiningOp<DecomposeBlockScaledOp>();
    if (!decompose || op.getData() != decompose.getData() ||
        op.getScale() != decompose.getScale() ||
        op.getType() != decompose.getInput().getType())
      return failure();

    const auto blockScaledType =
        cast<BlockScaledType>(cast<TensorType>(op.getType()).getElementType());
    if (op.getBlockShape() != blockScaledType.getBlockShape())
      return failure();

    rewriter.replaceOp(op, decompose.getInput());
    return success();
  }
};

struct PlanarComponentPair {
  Value data;
  Value scale;
};

LogicalResult checkFunctionArguments(ComposeBlockScaledOp compose) {
  auto function = compose->getParentOfType<func::FuncOp>();
  if (!function)
    return compose.emitOpError("must be inside a function");

  for (Value operand : compose.getOperands()) {
    auto argument = dyn_cast<BlockArgument>(operand);
    if (!argument || argument.getOwner() != &function.getBody().front())
      return compose.emitOpError(
          "requires data and scale operands to be function entry arguments");
  }
  return success();
}

LogicalResult checkFunctionReturns(DecomposeBlockScaledOp decompose) {
  auto function = decompose->getParentOfType<func::FuncOp>();
  if (!function)
    return decompose.emitOpError("must be inside a function");

  for (Value result : decompose.getResults()) {
    if (result.use_empty())
      return decompose.emitOpError(
          "requires both data and scale results to be returned");
    for (Operation *user : result.getUsers()) {
      auto returnOp = dyn_cast<func::ReturnOp>(user);
      if (!returnOp || returnOp->getParentOp() != function)
        return decompose.emitOpError(
            "requires data and scale results to be used only by returns of "
            "the enclosing function");
    }
  }
  return success();
}

// This analysis is expected to be run after decompose_block_scaled ->
// compose_block_scaled sequences have been folded. Therefore, it's expected
// compose_block_scaled operations exist only at function argument boundaries.
FailureOr<SmallVector<PlanarComponentPair>>
getPlanarArgumentComponentPairs(ModuleOp module) {
  SmallVector<PlanarComponentPair> planarComponentPairs;
  WalkResult result =
      module.walk([&planarComponentPairs](ComposeBlockScaledOp op) {
        if (failed(checkFunctionArguments(op)))
          return WalkResult::interrupt();
        planarComponentPairs.push_back({op.getData(), op.getScale()});
        return WalkResult::advance();
      });

  if (result.wasInterrupted())
    return failure();

  return planarComponentPairs;
}

// This analysis is expected to be run after decompose_block_scaled ->
// compose_block_scaled sequences have been folded. Therefore, it's expected
// decompose_block_scaled operations exist only at function result boundaries.
FailureOr<SmallVector<PlanarComponentPair>>
getPlanarResultComponentPairs(ModuleOp module) {
  SmallVector<PlanarComponentPair> planarComponentPairs;
  WalkResult result =
      module.walk([&planarComponentPairs](DecomposeBlockScaledOp op) {
        if (failed(checkFunctionReturns(op)))
          return WalkResult::interrupt();
        planarComponentPairs.push_back({op->getResult(0), op.getResult(1)});
        return WalkResult::advance();
      });

  if (result.wasInterrupted())
    return failure();

  return planarComponentPairs;
}

LogicalResult rewriteFunctionArguments(
    func::FuncOp func,
    const SmallVector<PlanarComponentPair> &planarArgumentPairs) {
  for (const auto &pair : planarArgumentPairs) {
    const auto dataArgument = cast<BlockArgument>(pair.data);
    const auto scaleArgument = cast<BlockArgument>(pair.scale);
    if (!dataArgument.hasOneUse() || !scaleArgument.hasOneUse())
      return func.emitOpError(
          "requires planar arguments to be used by a single compose");

    auto compose = dyn_cast<ComposeBlockScaledOp>(*dataArgument.user_begin());
    if (!compose || compose.getData() != dataArgument ||
        compose.getScale() != scaleArgument)
      return func.emitOpError("requires a matching boundary compose");

    const unsigned argumentIndex = dataArgument.getArgNumber();
    if (failed(func.insertArgument(argumentIndex, compose.getType(), {},
                                   compose.getLoc())))
      return failure();
    compose.getResult().replaceAllUsesWith(func.getArgument(argumentIndex));
    compose.erase();

    llvm::BitVector argumentsToErase(func.getNumArguments());
    argumentsToErase.set(dataArgument.getArgNumber());
    argumentsToErase.set(scaleArgument.getArgNumber());
    if (failed(func.eraseArguments(argumentsToErase)))
      return failure();
  }

  return success();
}

LogicalResult rewriteFunctionResult(
    func::FuncOp func,
    const SmallVector<PlanarComponentPair> &planarResultPairs) {
  if (planarResultPairs.empty())
    return success();

  auto returns = func.getBody().getOps<func::ReturnOp>();
  if (!llvm::hasSingleElement(returns))
    return func.emitOpError(
        "requires a single function return when rewriting planar results");
  auto returnOp = *returns.begin();

  for (const PlanarComponentPair &pair : planarResultPairs) {
    auto decompose = pair.data.getDefiningOp<DecomposeBlockScaledOp>();
    if (!decompose || decompose.getData() != pair.data ||
        decompose.getScale() != pair.scale)
      return func.emitOpError("requires a matching boundary decompose");
    if (!pair.data.hasOneUse() || !pair.scale.hasOneUse() ||
        pair.data.use_begin()->getOwner() != returnOp.getOperation() ||
        pair.scale.use_begin()->getOwner() != returnOp.getOperation())
      return returnOp.emitOpError(
          "requires each planar result pair to be returned exactly once");

    const unsigned dataIndex = pair.data.use_begin()->getOperandNumber();
    const unsigned scaleIndex = pair.scale.use_begin()->getOperandNumber();
    Value packed = decompose.getInput();

    // Insert the newly formed packed result at the index of the data tensor
    if (failed(func.insertResult(dataIndex, packed.getType(), {})) ||
        failed(func.eraseResult(dataIndex + 1)) ||
        failed(func.eraseResult(scaleIndex)))
      return failure();

    returnOp->setOperand(dataIndex, packed);
    returnOp->eraseOperand(scaleIndex);
    decompose.erase();
  }
  return success();
}
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
    target.addDynamicallyLegalOp<ConstOp>([&converter](ConstOp op) {
      return converter.isLegal(op.getOperation());
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
    patterns.add<ConvertTosaOpToPlanarAPI, ConvertConstToPlanar>(converter,
                                                                 context);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

class TosaRemovePlanarBlockScaledPass
    : public impl::TosaRemovePlanarBlockScaledPassBase<
          TosaRemovePlanarBlockScaledPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Fold decompose_block_scaled -> compose_block_scaled sequences
    RewritePatternSet patterns(&getContext());
    patterns
        .add<RemoveDecomposeComposeBlockScaled, ComposeBlockScaledConstants>(
            &getContext());
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      return signalPassFailure();

    if (convertFunctionBoundaries) {
      FailureOr<SmallVector<PlanarComponentPair>> maybeArgumentPairs =
          getPlanarArgumentComponentPairs(getOperation());
      if (failed(maybeArgumentPairs))
        return signalPassFailure();
      for (const auto &pair : *maybeArgumentPairs) {
        auto function = cast<func::FuncOp>(
            cast<BlockArgument>(pair.data).getOwner()->getParentOp());
        if (failed(rewriteFunctionArguments(function, {pair})))
          return signalPassFailure();
      }
      FailureOr<SmallVector<PlanarComponentPair>> maybeResultPairs =
          getPlanarResultComponentPairs(module);
      if (failed(maybeResultPairs))
        return signalPassFailure();

      for (const auto &pair : *maybeResultPairs) {
        auto function =
            pair.data.getDefiningOp()->getParentOfType<func::FuncOp>();
        if (failed(rewriteFunctionResult(function, {pair})))
          return signalPassFailure();
      }
    }

    // Check all compose_block_scaled and decompose_block_scaled operations have
    // been removed
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<ComposeBlockScaledOp, DecomposeBlockScaledOp>();
    if (failed(applyFullConversion(getOperation(), target,
                                   RewritePatternSet(&getContext()))))
      signalPassFailure();
  }
};
} // namespace tosa
} // namespace mlir
