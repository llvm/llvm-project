//===- TosaOps.cpp - MLIR Dialect for TOSA --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the TOSA Specification:
// https://developer.mlplatform.org/w/tosa/
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::tosa;

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Tosa dialect structs and interface includes.
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaStructs.cpp.inc"

namespace {
//===----------------------------------------------------------------------===//
// Dialect Function Inliner Interface.
//===----------------------------------------------------------------------===//
struct TosaInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks.
  //===--------------------------------------------------------------------===//

  /// All operations can be inlined by default.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       BlockAndValueMapping &map) const final {
    return true;
  }

  /// All regions with If and While parent operators can be inlined.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &map) const final {
    return (isa<tosa::IfOp>(dest->getParentOp()) ||
            isa<tosa::WhileOp>(dest->getParentOp()));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// TOSA control flow support.
//===----------------------------------------------------------------------===//

/// Returns the while loop body.
Region &tosa::WhileOp::getLoopBody() { return body(); }

bool tosa::WhileOp::isDefinedOutsideOfLoop(Value value) {
  return !body().isAncestor(value.getParentRegion());
}

LogicalResult WhileOp::moveOutOfLoop(ArrayRef<mlir::Operation *> ops) {
  if (ops.empty())
    return success();

  Operation *tosaWhileOp = this->getOperation();
  for (auto *op : ops)
    op->moveBefore(tosaWhileOp);

  return success();
}

//===----------------------------------------------------------------------===//
// Tosa dialect initialization.
//===----------------------------------------------------------------------===//

void TosaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
      >();
  addInterfaces<TosaInlinerInterface>();
}

Operation *TosaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Tosa dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (value.isa<ElementsAttr>())
    return builder.create<tosa::ConstOp>(loc, type, value.cast<ElementsAttr>());
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Operator Canonicalizers.
//===----------------------------------------------------------------------===//

struct ConcatOptimization : public OpRewritePattern<tosa::ConcatOp> {
  using OpRewritePattern<tosa::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ConcatOp op,
                                PatternRewriter &rewriter) const override {
    if (op.input1().size() != 1)
      return failure();
    if (op.input1().front().getType() != op.getType()) {
      rewriter
          .replaceOpWithNewOp<tensor::CastOp>(op, op.getType(),
                                              op.input1().front())
          .getResult();
      return success();
    }

    rewriter.replaceOp(op, op.input1().front());
    return success();
  }
};

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConcatOptimization>(context);
}

struct ReshapeReshapeOptimization : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input1();
    Operation *definingOp = input.getDefiningOp();
    if (!definingOp)
      return failure();

    if (tosa::ReshapeOp reshapeOp = dyn_cast<tosa::ReshapeOp>(definingOp)) {
      rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
          op, op.getType(), reshapeOp.input1(), op.new_shape());
      return success();
    }

    return failure();
  }
};

struct ReshapeConstOptimization : public OpRewritePattern<tosa::ReshapeOp> {
  using OpRewritePattern<tosa::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input1();
    ArrayAttr newShape = op.new_shape();

    // Check if input is constant
    DenseElementsAttr inputAttr;
    if (!matchPattern(input, m_Constant(&inputAttr)))
      return failure();

    // Check if has >1 consumer and is not splat
    if (!input.hasOneUse() && !inputAttr.isSplat())
      return failure();

    // Grab the new shape
    SmallVector<int64_t> newShapeValues = llvm::to_vector<6>(
        llvm::map_range(newShape.getValue(), [](const Attribute &val) {
          return val.cast<IntegerAttr>().getValue().getSExtValue();
        }));

    // Build new const op with correct output shape
    ShapedType inputShape = input.getType().cast<ShapedType>();
    DenseElementsAttr outputAttr =
        inputAttr.reshape(inputShape.clone(newShapeValues));
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputAttr.getType(),
                                               outputAttr);
    return success();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<ReshapeReshapeOptimization>(context);
  results.insert<ReshapeConstOptimization>(context);
}

struct ConstantTransposeOptimization
    : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = op.getType().cast<ShapedType>();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    // TOSA supports quantized types.
    if (!outputType.getElementType().isIntOrIndexOrFloat())
      return failure();

    DenseElementsAttr inputValues;
    if (!matchPattern(op.input1(), m_Constant(&inputValues)))
      return failure();
    // Make sure the input is a constant that has a single user.
    if (!llvm::hasSingleElement(op.input1().getDefiningOp()->getUsers()))
      return failure();

    DenseIntElementsAttr permAttr;
    if (!matchPattern(op.perms(), m_Constant(&permAttr)))
      return failure();
    auto permValues = llvm::to_vector<6>(llvm::map_range(
        // TOSA allows both 32- and 64-bit integer tensors here.
        permAttr.getValues<APInt>(),
        [](const APInt &val) { return val.getZExtValue(); }));

    auto inputType = op.input1().getType().cast<ShapedType>();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t numElements = inputType.getNumElements();

    SmallVector<Attribute, 4> outputValues;
    outputValues.resize(numElements);

    // Transpose the input constant. Because we don't know its rank in advance,
    // we need to loop over the range [0, element count) and delinearize the
    // index.
    auto attrValues = inputValues.getValues<Attribute>();
    for (int srcLinearIndex = 0; srcLinearIndex < numElements;
         ++srcLinearIndex) {
      SmallVector<uint64_t, 6> srcIndices(inputType.getRank(), 0);
      int totalCount = srcLinearIndex;
      for (int dim = inputType.getRank() - 1; dim >= 0; --dim) {
        srcIndices[dim] = totalCount % inputShape[dim];
        totalCount /= inputShape[dim];
      }

      SmallVector<uint64_t, 6> dstIndices(outputType.getRank(), 0);
      for (int dim = outputType.getRank() - 1; dim >= 0; --dim)
        dstIndices[dim] = srcIndices[permValues[dim]];

      uint64_t dstLinearIndex = dstIndices.front();
      for (int dim = 1; dim < outputType.getRank(); ++dim)
        dstLinearIndex = dstLinearIndex * outputShape[dim] + dstIndices[dim];

      outputValues[dstLinearIndex] = attrValues[srcIndices];
    }

    rewriter.replaceOpWithNewOp<tosa::ConstOp>(
        op, outputType, DenseElementsAttr::get(outputType, outputValues));
    return success();
  }
};

struct NoOpOptimization : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto perm = op.perms();

    DenseIntElementsAttr permAttr;
    if (!matchPattern(perm, m_Constant(&permAttr))) {
      return failure();
    }

    SmallVector<int64_t> permValues = llvm::to_vector<6>(
        llvm::map_range(permAttr.getValues<APInt>(),
                        [](const APInt &val) { return val.getSExtValue(); }));

    for (int i = 0, s = permValues.size(); i < s; i++) {
      if (i != permValues[i]) {
        return failure();
      }
    }

    rewriter.replaceOp(op, op.input1());
    return success();
  }
};

void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<ConstantTransposeOptimization>(context);
  results.insert<NoOpOptimization>(context);
}

struct AddZeroOptimization : public OpRewritePattern<tosa::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto input1 = op.input1();
    auto input2 = op.input2();

    DenseElementsAttr input1Attr;
    if (matchPattern(input1, m_Constant(&input1Attr)) && input1Attr.isSplat() &&
        input2.getType() == op.getType()) {
      if (input1Attr.getType().getElementType().isa<IntegerType>() &&
          input1Attr.getSplatValue<APInt>().isZero()) {
        rewriter.replaceOp(op, op.input2());
        return success();
      }
    }

    DenseElementsAttr input2Attr;
    if (matchPattern(input2, m_Constant(&input2Attr)) && input2Attr.isSplat() &&
        input1.getType() == op.getType()) {
      if (input2Attr.getType().getElementType().isa<IntegerType>() &&
          input2Attr.getSplatValue<APInt>().isZero()) {
        rewriter.replaceOp(op, op.input1());
        return success();
      }
    }

    return failure();
  }
};

void AddOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<AddZeroOptimization>(context);
}

struct MulOneOptimization : public OpRewritePattern<tosa::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto input1 = op.input1();
    auto input2 = op.input2();

    DenseElementsAttr input1Attr;
    if (matchPattern(input1, m_Constant(&input1Attr)) && input1Attr.isSplat() &&
        input2.getType() == op.getType()) {
      if (input1Attr.getType().getElementType().isa<FloatType>() &&
          input1Attr.getSplatValue<APFloat>().isExactlyValue(1)) {
        rewriter.replaceOp(op, op.input2());
        return success();
      }

      if (input1Attr.getType().getElementType().isa<IntegerType>() &&
          matchPattern(input1, m_One())) {
        rewriter.replaceOp(op, op.input2());
        return success();
      }
    }

    DenseElementsAttr input2Attr;
    if (matchPattern(input2, m_Constant(&input2Attr)) && input2Attr.isSplat() &&
        input1.getType() == op.getType()) {
      if (input2Attr.getType().getElementType().isa<FloatType>() &&
          input2Attr.getSplatValue<APFloat>().isExactlyValue(1)) {
        rewriter.replaceOp(op, op.input1());
        return success();
      }

      if (input2Attr.getType().getElementType().isa<IntegerType>() &&
          matchPattern(input2, m_One())) {
        rewriter.replaceOp(op, op.input1());
        return success();
      }
    }

    return failure();
  }
};

void MulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MulOneOptimization>(context);
}

struct MaterializePadValue : public OpRewritePattern<tosa::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::PadOp op,
                                PatternRewriter &rewriter) const override {
    if (op.pad_const())
      return failure();

    auto input = op.input1();
    auto padding = op.padding();

    ShapedType inputTy = input.getType().cast<ShapedType>();
    Type elementTy = inputTy.getElementType();

    Attribute constantAttr;
    if (elementTy.isa<FloatType>())
      constantAttr = rewriter.getFloatAttr(elementTy, 0.0);
    else if (elementTy.isa<IntegerType>() && !op.quantization_info())
      constantAttr = rewriter.getIntegerAttr(elementTy, 0);
    else if (elementTy.isa<IntegerType>() && op.quantization_info()) {
      auto value = op.quantization_info().getValue().input_zp().getValue();
      constantAttr = rewriter.getIntegerAttr(elementTy, value.getZExtValue());
    }

    if (!constantAttr) {
      return rewriter.notifyMatchFailure(
          op,
          "tosa.pad to linalg lowering encountered an unknown element type");
    }

    auto denseAttr = DenseElementsAttr::get(
        RankedTensorType::get({}, elementTy), constantAttr);
    auto constantVal = rewriter.create<tosa::ConstOp>(
        op.getLoc(), denseAttr.getType(), denseAttr);

    rewriter.replaceOpWithNewOp<tosa::PadOp>(
        op, op.getType(), ValueRange{input, padding, constantVal},
        op->getAttrs());
    return success();
  }
};

void PadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<MaterializePadValue>(context);
}

struct MaxPool2dIsNoOp : public OpRewritePattern<tosa::MaxPool2dOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::MaxPool2dOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    Value output = op.output();
    ShapedType inputType = input.getType().cast<ShapedType>();
    ShapedType outputType = output.getType().cast<ShapedType>();

    if (!inputType.hasStaticShape() || !outputType.hasStaticShape()) {
      return failure();
    }

    // If the output and input shapes are 1x1, then this is a no op.
    ArrayRef<int64_t> outputShape = outputType.getShape();
    if (outputShape[1] != 1 || outputShape[2] != 1) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    if (inputShape[1] != 1 || inputShape[2] != 1) {
      return failure();
    }

    rewriter.replaceOp(op, input);
    return success();
  }
};

void MaxPool2dOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<MaxPool2dIsNoOp>(context);
}

struct ClampIsNoOp : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();
    auto inputType = op.input().getType().template dyn_cast<RankedTensorType>();
    auto inputElementType = inputType.getElementType();

    if (!inputType.hasStaticShape()) {
      return failure();
    }

    if (inputElementType.isF32()) {
      auto minClamp = op.min_fp();
      auto maxClamp = op.max_fp();
      bool isMin = (minClamp.isLargest() || minClamp.isInfinity()) &&
                   minClamp.isNegative();
      bool isMax = (maxClamp.isLargest() || maxClamp.isInfinity()) &&
                   !maxClamp.isNegative();

      if (isMin && isMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    if (inputElementType.isUnsignedInteger()) {
      int64_t minClamp = op.min_int();
      int64_t maxClamp = op.max_int();

      int64_t intMin =
          APInt::getMinValue(inputElementType.getIntOrFloatBitWidth())
              .getZExtValue();
      int64_t intMax =
          APInt::getMaxValue(inputElementType.getIntOrFloatBitWidth())
              .getZExtValue();

      if (minClamp <= intMin && maxClamp >= intMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    if (inputElementType.isa<IntegerType>()) {
      int64_t minClamp = op.min_int();
      int64_t maxClamp = op.max_int();

      int64_t intMin =
          APInt::getSignedMinValue(inputElementType.getIntOrFloatBitWidth())
              .getSExtValue();
      int64_t intMax =
          APInt::getSignedMaxValue(inputElementType.getIntOrFloatBitWidth())
              .getSExtValue();

      if (minClamp <= intMin && maxClamp >= intMax) {
        rewriter.replaceOp(op, input);
        return success();
      }
      return failure();
    }

    return failure();
  }
};

struct ClampClampOptimization : public OpRewritePattern<tosa::ClampOp> {
  using OpRewritePattern<tosa::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::ClampOp op,
                                PatternRewriter &rewriter) const override {
    Value input = op.input();

    Operation *definingOp = input.getDefiningOp();
    if (!definingOp)
      return failure();

    if (tosa::ClampOp clampOp = dyn_cast<tosa::ClampOp>(definingOp)) {
      auto min_fp = std::max(op.min_fp(), clampOp.min_fp()).convertToFloat();
      auto max_fp = std::min(op.max_fp(), clampOp.max_fp()).convertToFloat();

      auto min_int = std::max(op.min_int(), clampOp.min_int());
      auto max_int = std::min(op.max_int(), clampOp.max_int());

      rewriter.replaceOpWithNewOp<tosa::ClampOp>(
          op, op.getType(), clampOp.input(),
          rewriter.getI64IntegerAttr(min_int),
          rewriter.getI64IntegerAttr(max_int), rewriter.getF32FloatAttr(min_fp),
          rewriter.getF32FloatAttr(max_fp));
      return success();
    }

    return failure();
  }
};

void ClampOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.insert<ClampIsNoOp>(context);
  results.insert<ClampClampOptimization>(context);
}

//===----------------------------------------------------------------------===//
// Operator Folders.
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  if (input().getType() == getType())
    return input();
  return {};
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return valueAttr();
}

#define ReduceFolder(OP)                                                       \
  OpFoldResult OP::fold(ArrayRef<Attribute> operands) {                        \
    ShapedType inputTy = input().getType().cast<ShapedType>();                 \
    if (!inputTy.hasRank())                                                    \
      return {};                                                               \
    if (inputTy.getDimSize(axis()) == 1)                                       \
      return input();                                                          \
    return {};                                                                 \
  }

ReduceFolder(ReduceAllOp) ReduceFolder(ReduceAnyOp) ReduceFolder(ReduceMaxOp)
    ReduceFolder(ReduceMinOp) ReduceFolder(ReduceProdOp)
        ReduceFolder(ReduceSumOp)
#undef ReduceFolder

            OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  auto inputTy = input1().getType().dyn_cast<RankedTensorType>();
  auto outputTy = getType().dyn_cast<RankedTensorType>();

  if (!inputTy || !outputTy || inputTy != outputTy)
    return {};
  return input1();
}

OpFoldResult PadOp::fold(ArrayRef<Attribute> operands) {
  // If the pad is all zeros we can fold this operation away.
  if (operands[1]) {
    auto densePad = operands[1].cast<DenseElementsAttr>();
    if (densePad.isSplat() && densePad.getSplatValue<APInt>().isZero()) {
      return input1();
    }
  }

  return {};
}

OpFoldResult SliceOp::fold(ArrayRef<Attribute> operands) {
  auto inputTy = input().getType().dyn_cast<RankedTensorType>();
  auto outputTy = getType().dyn_cast<RankedTensorType>();

  if (!inputTy || !outputTy || inputTy != outputTy)
    return {};
  if (inputTy.hasStaticShape())
    return input();

  return {};
}

OpFoldResult TileOp::fold(ArrayRef<Attribute> operands) {
  bool allOnes = true;
  for (Attribute val : multiples().getValue()) {
    allOnes = allOnes && val.cast<IntegerAttr>().getValue().getSExtValue() == 1;
  }

  if (allOnes && input1().getType() == getType())
    return input1();
  return {};
}

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  if (!operands[1])
    return {};

  // Transposing splat values just means reshaping.
  if (auto input = operands[0].dyn_cast_or_null<DenseElementsAttr>()) {
    if (input.isSplat())
      return input.reshape(getType().cast<ShapedType>());
  }

  auto perms = llvm::to_vector<6>(llvm::map_range(
      operands[1].cast<DenseIntElementsAttr>().getValues<APInt>(),
      [](const APInt &val) { return val.getSExtValue(); }));

  if (llvm::equal(llvm::seq<int64_t>(0, perms.size()), perms) &&
      input1().getType() == getType())
    return input1();
  return {};
}

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult verifyConvOp(T op) {
  // All TOSA conv ops have an input() and weight().
  auto inputType = op.input().getType().template dyn_cast<RankedTensorType>();
  auto weightType = op.weight().getType().template dyn_cast<RankedTensorType>();

  // Must be ranked tensor types
  if (!inputType) {
    op.emitOpError("expect a ranked tensor for input, got ") << op.input();
    return failure();
  }
  if (!weightType) {
    op.emitOpError("expect a ranked tensor for weight, got ") << op.weight();
    return failure();
  }

  auto inputEType = inputType.getElementType();
  auto weightEType = weightType.getElementType();

  bool inputIsQuant = !inputEType.template isa<FloatType>();
  bool weightIsQuant = !weightEType.template isa<FloatType>();

  // Either both must be quantized or both unquantized.
  if (inputIsQuant != weightIsQuant) {
    op.emitOpError(
        "expect both input and weight to be float or not together, got ")
        << inputEType << " and " << weightEType;
    return failure();
  }

  // Quantized type must have constructed the quantizationattr, and unquantized
  // types should not have a quantizationattr.
  if ((inputIsQuant && !op.quantization_info()) ||
      (!inputIsQuant && op.quantization_info())) {
    op.emitOpError("quantizationattr is required for quantized type, and not "
                   "allowed for float type");
    return failure();
  }

  return success();
}

static LogicalResult verifyAveragePoolOp(tosa::AvgPool2dOp op) {
  auto inputETy = op.input().getType().cast<ShapedType>().getElementType();
  auto resultETy = op.getType().cast<ShapedType>().getElementType();

  if (auto quantType = inputETy.dyn_cast<mlir::quant::UniformQuantizedType>())
    inputETy = quantType.getStorageType();

  if (auto quantType = resultETy.dyn_cast<mlir::quant::UniformQuantizedType>())
    resultETy = quantType.getStorageType();

  if (inputETy.isF32() && resultETy.isF32())
    return success();
  if (inputETy.isInteger(8) && resultETy.isInteger(8))
    return success();
  if (inputETy.isInteger(16) && resultETy.isInteger(16))
    return success();

  return op.emitOpError("input/output element types are incompatible.");
}

//===----------------------------------------------------------------------===//
// TOSA Operator Quantization Builders.
//===----------------------------------------------------------------------===//

/// This builder is called on all convolution operators except TransposeConv,
/// which has specialized output shape semantics. The builder also defines the
/// bitwidth of the output given the bit width of the input & weight content.
static void buildConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                     Type outputType, Value input, Value weight,
                                     Value bias, ArrayAttr pad,
                                     ArrayAttr stride, ArrayAttr dilation) {

  result.addOperands({input, weight, bias});
  result.addAttribute("pad", pad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);

  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// Handles tosa.transpose_conv2d which has outpad and output shape attributes.
static void
buildTransConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                              Type outputType, Value input, Value weight,
                              Value bias, ArrayAttr outpad, ArrayAttr stride,
                              ArrayAttr dilation, ArrayAttr outputShape) {
  result.addOperands({input, weight, bias});
  result.addAttribute("out_pad", outpad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);
  result.addAttribute("out_shape", outputShape);
  auto quantAttr = ::buildConvOpQuantizationAttr(builder, input, weight);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// The tosa.fully_connected op has its own builder as it does not have
/// strides/dilation/padding.
static void buildFCOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                   Type outputType, Value input, Value weight,
                                   Value bias) {

  result.addOperands({input, weight, bias});
  auto quantAttr = ::buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);
    result.addTypes(
        buildConvOpResultTypeInfo(builder, outputType, input, weight));
  } else {
    result.addTypes(outputType);
  }
}

/// The tosa.matmul op is also intended to be generated where a fully_connected
/// op must be constructed where the weight is not a constant. In this case,
/// the fully_connected op must be expressed using matmul.
/// TODO: Add link to the leglization document explaining this.
static void buildMatMulOpWithQuantInfo(OpBuilder &builder,
                                       OperationState &result, Type outputType,
                                       Value a, Value b) {
  result.addOperands({a, b});
  auto quantAttr = ::buildMatMulOpQuantizationAttr(builder, a, b);

  if (quantAttr) {
    result.addAttribute("quantization_info", quantAttr);

    auto inputType = a.getType().dyn_cast<ShapedType>();
    assert(inputType && "Input must be a shaped tensor type!");

    auto inputQType = inputType.getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();
    assert(inputQType && "Tensor must have quantized datatype!");

    unsigned inputBits = inputQType.getStorageTypeIntegralWidth();

    auto outputShapedType = outputType.dyn_cast<ShapedType>();
    assert(outputShapedType && "Output must be a shaped type");

    IntegerType accElementType;
    if (inputBits == 16)
      accElementType = builder.getIntegerType(48);
    else
      accElementType = builder.getI32Type();
    auto accType = outputShapedType.clone(accElementType);
    result.addTypes(accType);
  } else {
    result.addTypes(outputType);
  }
}

/// Both the tosa.avg_pool2d and unary ops use the same UnaruOpQuantizationAttr
/// but avg_pool operator has its own builder as it has additional parameters
/// not part of the unary ops.
static void buildAvgPool2dOpWithQuantInfo(OpBuilder &builder,
                                          OperationState &result,
                                          Type outputType, Value input,
                                          ArrayAttr kernel, ArrayAttr stride,
                                          ArrayAttr pad) {
  result.addOperands(input);
  result.addAttribute("kernel", kernel);
  result.addAttribute("stride", stride);
  result.addAttribute("pad", pad);
  auto quantAttr = buildUnaryOpQuantizationAttr(builder, input, outputType);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

/// This builder is called on single-parameter unary operators that have scale
/// relationship between their input and output, expressed by the
/// UnaryOpQuantizationAttr.
static void buildUnaryOpWithQuantInfo(OpBuilder &builder,
                                      OperationState &result, Type outputType,
                                      Value input) {
  result.addOperands(input);
  auto quantAttr = buildUnaryOpQuantizationAttr(builder, input, outputType);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

/// This builder is called on TOSA pad operator that needs to create its own
/// OptionalAttr quantization_attr parameter to scale the padding values
/// correctly. No pad_const is interpreted as zero-padding.
static void buildPadOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                    Type outputType, Value input,
                                    Value paddings) {
  result.addOperands({input, paddings});
  auto quantAttr = buildPadOpQuantizationAttr(builder, input);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

/// This builder is called on TOSA pad operator when an explicit pad_const
/// value is passed in. It also optionally constructs quantization_attr.
static void buildExplicitValuePadOpWithQuantInfo(OpBuilder &builder,
                                                 OperationState &result,
                                                 Type outputType, Value input,
                                                 Value paddings,
                                                 Value padConst) {
  result.addOperands({input, paddings, padConst});
  auto quantAttr = buildPadOpQuantizationAttr(builder, input);
  if (quantAttr)
    result.addAttribute("quantization_info", quantAttr);
  result.types.push_back(outputType);
}

//===----------------------------------------------------------------------===//
// TOSA Operator Return Type Inference.
//===----------------------------------------------------------------------===//

static void getI64Values(ArrayAttr arrayAttr, SmallVector<int64_t> &values) {
  for (auto it : arrayAttr) {
    values.push_back(it.cast<IntegerAttr>().getValue().getSExtValue());
  }
}

static void getF64Values(ArrayAttr arrayAttr, SmallVector<double> &values) {
  for (auto it : arrayAttr) {
    values.push_back(it.cast<FloatAttr>().getValueAsDouble());
  }
}

LogicalResult tosa::ArgMaxOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  IntegerAttr axis = attributes.get("axis").cast<IntegerAttr>();
  int32_t axisVal = axis.getValue().getSExtValue();

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  SmallVector<int64_t> outShape;
  outShape.reserve(inputShape.getRank() - 1);
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (i == axisVal)
      continue;
    outShape.push_back(inputShape.getDimSize(i));
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::ConcatOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Infer all dimension sizes by reducing based on inputs.
  int32_t axis =
      attributes.get("axis").cast<IntegerAttr>().getValue().getSExtValue();
  llvm::SmallVector<int64_t> outputShape;
  bool hasRankedInput = false;
  for (auto operand : operands) {
    ShapeAdaptor operandShape = operands.getShape(operand);
    if (!operandShape.hasRank())
      continue;

    // Copy the Operand's rank.
    if (!hasRankedInput)
      outputShape.resize(operandShape.getRank(), ShapedType::kDynamicSize);

    // Copy shapes until the dim is non-dynamic.
    for (int i = 0, s = operandShape.getRank(); i < s; i++) {
      if (i == axis || operandShape.isDynamicDim(i))
        continue;
      if (outputShape[i] == ShapedType::kDynamicSize)
        outputShape[i] = operandShape.getDimSize(i);
      if (outputShape[i] != operandShape.getDimSize(i))
        return failure();
    }

    hasRankedInput = true;
  }

  if (!hasRankedInput) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // Determine the dimension size along the concatenation axis.
  int concatDimSize = 0;
  for (auto operand : operands) {
    ShapeAdaptor operandShape = operands.getShape(operand);

    // We need to know the length of the concatenation axis of all inputs to
    // determine the dimension size of the output shape.
    if (!operandShape.hasRank() || operandShape.isDynamicDim(axis)) {
      concatDimSize = ShapedType::kDynamicSize;
      break;
    }

    concatDimSize += operandShape.getDimSize(axis);
  }

  outputShape[axis] = concatDimSize;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::FullyConnectedOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor weightShape = operands.getShape(1);
  ShapeAdaptor biasShape = operands.getShape(2);

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(2, ShapedType::kDynamicSize);

  if (inputShape.hasRank()) {
    outShape[0] = inputShape.getDimSize(0);
  }

  if (weightShape.hasRank()) {
    outShape[1] = weightShape.getDimSize(0);
  }

  if (biasShape.hasRank()) {
    outShape[1] = outShape[1] == ShapedType::kDynamicSize
                      ? biasShape.getDimSize(0)
                      : outShape[1];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor lhsShape = operands.getShape(0);
  ShapeAdaptor rhsShape = operands.getShape(1);

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(3, ShapedType::kDynamicSize);

  if (lhsShape.hasRank()) {
    outShape[0] = lhsShape.getDimSize(0);
    outShape[1] = lhsShape.getDimSize(1);
  }

  if (rhsShape.hasRank()) {
    outShape[0] = outShape[0] == ShapedType::kDynamicSize
                      ? rhsShape.getDimSize(0)
                      : outShape[0];
    outShape[2] = rhsShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::PadOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor paddingShape = operands.getShape(1);
  SmallVector<int64_t> outputShape;

  // If both inputs have unknown shape, we cannot determine the shape of the
  // output.
  if (!inputShape.hasRank() && !paddingShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // If the input rank is unknown we can info the output rank using the padding
  // shape's first dim.
  if (!inputShape.hasRank()) {
    if (paddingShape.isDynamicDim(0)) {
      inferredReturnShapes.push_back(ShapedTypeComponents());
      return success();
    }

    outputShape.resize(paddingShape.getDimSize(0), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  DenseIntElementsAttr paddings;
  // If the paddings value is not a constant, all dimensions must be dynamic.
  if (!matchPattern(operands[1], m_Constant(&paddings))) {
    outputShape.resize(inputShape.getRank(), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  SmallVector<int64_t> paddingValues;
  for (auto val : paddings) {
    paddingValues.push_back(val.getSExtValue());
  }

  outputShape.reserve(inputShape.getRank());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.isDynamicDim(i)) {
      outputShape.push_back(ShapedType::kDynamicSize);
      continue;
    }

    outputShape.push_back(inputShape.getDimSize(i) + paddingValues[i * 2] +
                          paddingValues[i * 2 + 1]);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::SliceOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ArrayAttr sizes = SliceOpAdaptor(operands, attributes).size();
  SmallVector<int64_t> outputShape;
  outputShape.reserve(sizes.size());
  for (auto val : sizes) {
    outputShape.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::TableOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.resize(1);
  inputShape.getDims(inferredReturnShapes[0]);
  return success();
}

LogicalResult tosa::TileOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TileOpAdaptor adaptor(operands, attributes);
  ArrayAttr multiples = adaptor.multiples();
  ShapeAdaptor inputShape = operands.getShape(0);
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(multiples.size(), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // We need the multiple values to determine the output shape.
  SmallVector<int64_t> multipleValues;
  multipleValues.reserve(multiples.size());
  for (auto val : multiples) {
    multipleValues.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
  }

  // Any non dynamic dimension can be multiplied to a known size.
  outputShape.reserve(multiples.size());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    int dim = inputShape.getDimSize(i);
    if (dim != ShapedType::kDynamicSize)
      dim *= multipleValues[i];
    outputShape.push_back(dim);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ReshapeOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ReshapeOpAdaptor adaptor(operands, attributes);
  ShapeAdaptor inputShape = operands.getShape(0);

  ArrayAttr newShape = adaptor.new_shape();
  llvm::SmallVector<int64_t> newShapeValue;
  getI64Values(newShape, newShapeValue);

  // We cannot infer from the total number of elements so we must take the
  // shape attribute as exact.
  if (!inputShape.hasRank() || !inputShape.hasStaticShape()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(newShapeValue));
    return success();
  }

  // Determine the number of elements covered by the slice of all static
  // dimensions. This allows us to infer the length of the remaining dynamic
  // dimension.
  int64_t numElements = inputShape.getNumElements();
  int64_t staticMul = 1;
  for (auto val : newShapeValue) {
    if (val != ShapedType::kDynamicSize) {
      staticMul *= val;
    }
  }

  // Determine the length of the dynamic dimension.
  for (auto &val : newShapeValue) {
    if (val == ShapedType::kDynamicSize)
      val = numElements / staticMul;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(newShapeValue));
  return success();
}

LogicalResult tosa::TransposeOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  ShapeAdaptor permsShape = operands.getShape(1);

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank() || !permsShape.hasRank() ||
      permsShape.isDynamicDim(0)) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  // This would imply the number of permutations does not match the rank of the
  // input which is illegal.
  if (permsShape.getDimSize(0) != inputShape.getRank()) {
    return failure();
  }

  // Without the input dims we cannot determine the output dim sizes but we
  // can determine the output rank.
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(permsShape.getDimSize(0), ShapedType::kDynamicSize);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Rank-0 means no permutations matter.
  if (inputShape.getRank() == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputShape.getRank(), inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputShape.getRank(), ShapedType::kDynamicSize);
  // If the permuations are a constant we can directly determine the output
  // shape.
  if (ShapeAdaptor permShape = operands.getValueAsShape(1)) {
    outputShape.reserve(inputShape.getRank());
    for (int i = 0, s = inputShape.getRank(); i < s; i++) {
      outputShape[i] = inputShape.getDimSize(permShape.getDimSize(i));
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::GatherOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamicSize);

  ShapeAdaptor valuesShape = operands.getShape(0);
  if (valuesShape.hasRank()) {
    outputShape[0] = valuesShape.getDimSize(0);
    outputShape[2] = valuesShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape = operands.getShape(1);
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamicSize)
      outputShape[0] = indicesShape.getDimSize(0);
    if (outputShape[1] == ShapedType::kDynamicSize)
      outputShape[1] = indicesShape.getDimSize(1);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ResizeOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ResizeOpAdaptor adaptor(operands, attributes);
  llvm::SmallVector<int64_t, 4> outputShape;
  outputShape.resize(4, ShapedType::kDynamicSize);

  int32_t inHeight = ShapedType::kDynamicSize;
  int32_t inWidth = ShapedType::kDynamicSize;

  ShapeAdaptor inputShape = operands.getShape(adaptor.input());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    outputShape[3] = inputShape.getDimSize(3);

    inHeight = inputShape.getDimSize(1);
    inWidth = inputShape.getDimSize(2);
  }

  int32_t shift = adaptor.shift();
  llvm::SmallVector<int64_t> newShape;
  getI64Values(adaptor.output_size(), newShape);
  outputShape[1] = newShape[0];
  outputShape[2] = newShape[1];

  llvm::SmallVector<int64_t> strideInt;
  llvm::SmallVector<int64_t> offsetInt;
  llvm::SmallVector<double> strideFp;
  llvm::SmallVector<double> offsetFp;
  getI64Values(adaptor.offset(), offsetInt);
  getF64Values(adaptor.offset_fp(), offsetFp);
  getI64Values(adaptor.stride(), strideInt);
  getF64Values(adaptor.stride_fp(), strideFp);

  // If we have a 0 zero in integers we know that the resize indexing needs to
  // be performed in floating point. Use the floating point varient to compute
  // the resize shape.
  bool fpMode = strideInt[0] == 0;

  // We can compute the output shape if attribute specifies unknown dimensions
  // based on the offset and stride. If we perfectly line up to the last index
  // we need to round up the size to include it.
  if (outputShape[1] == ShapedType::kDynamicSize && inHeight >= 0 && fpMode) {
    float sizeFp = (inHeight - offsetFp[0] - 1) / strideFp[0];
    float round = std::floor(sizeFp) == sizeFp ? 1 : 0;
    outputShape[1] = std::ceil(sizeFp) + round;
  }

  if (outputShape[2] == ShapedType::kDynamicSize && inWidth >= 0 && fpMode) {
    float sizeFp = (inWidth - offsetFp[1] - 1) / strideFp[1];
    float round = std::floor(sizeFp) == sizeFp ? 1 : 0;
    outputShape[2] = std::ceil(sizeFp) + round;
  }

  if (outputShape[1] == ShapedType::kDynamicSize && inHeight >= 0 && !fpMode) {
    int64_t size = (inHeight - 1);
    size = ((size << shift) - offsetInt[0]) / strideInt[0];
    outputShape[1] = size + 1;
  }

  if (outputShape[2] == ShapedType::kDynamicSize && inWidth >= 0 && !fpMode) {
    int64_t size = (inWidth - 1);
    size = ((size << shift) - offsetInt[1]) / strideInt[1];
    outputShape[2] = size + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ScatterOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamicSize);

  ShapeAdaptor valuesInShape = operands.getShape(0);
  if (valuesInShape.hasRank()) {
    outputShape[0] = valuesInShape.getDimSize(0);
    outputShape[1] = valuesInShape.getDimSize(1);
    outputShape[2] = valuesInShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape = operands.getShape(1);
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamicSize)
      outputShape[0] = indicesShape.getDimSize(0);
  }

  ShapeAdaptor inputShape = operands.getShape(2);
  if (inputShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamicSize)
      outputShape[0] = inputShape.getDimSize(0);
    if (outputShape[2] == ShapedType::kDynamicSize)
      outputShape[2] = inputShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

static LogicalResult ReduceInferReturnTypes(
    ShapeAdaptor operandShape, IntegerAttr axis,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  if (!operandShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  SmallVector<int64_t> outputShape;
  operandShape.getDims(outputShape);
  int64_t axisVal = axis.getValue().getSExtValue();
  outputShape[axisVal] = 1;
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

#define REDUCE_SHAPE_INFER(OP)                                                 \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::llvm::Optional<Location> location,               \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      RegionRange regions,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return ReduceInferReturnTypes(operands.getShape(0),                        \
                                  attributes.get("axis").cast<IntegerAttr>(),  \
                                  inferredReturnShapes);                       \
  }

REDUCE_SHAPE_INFER(tosa::ReduceAllOp)
REDUCE_SHAPE_INFER(tosa::ReduceAnyOp)
REDUCE_SHAPE_INFER(tosa::ReduceMaxOp)
REDUCE_SHAPE_INFER(tosa::ReduceMinOp)
REDUCE_SHAPE_INFER(tosa::ReduceProdOp)
REDUCE_SHAPE_INFER(tosa::ReduceSumOp)
#undef REDUCE_SHAPE_INFER

static LogicalResult resolveBroadcastShape(const ValueShapeRange &operands,
                                           SmallVector<int64_t> &outShape) {
  int64_t outRank = 0;
  for (int i = 0, e = operands.size(); i != e; ++i) {
    auto shape = operands.getShape(i);
    if (!shape.hasRank()) {
      return failure();
    }
    outRank = std::max<int64_t>(outRank, shape.getRank());
  }

  outShape.resize(outRank, 1);

  for (int i = 0, e = operands.size(); i != e; ++i) {
    auto shape = operands.getShape(i);
    auto rankDiff = outShape.size() - shape.getRank();

    for (size_t i = 0, e = shape.getRank(); i < e; ++i) {
      auto dim1 = outShape[i + rankDiff];
      auto dim2 = shape.getDimSize(i);
      auto resolvedDim = dim1;

      if (dim1 == 1) {
        resolvedDim = dim2;
      } else if (dim2 == 1) {
        resolvedDim = dim1;
      } else if (dim1 != dim2) {
        return failure();
      }
      outShape[i + rankDiff] = resolvedDim;
    }
  }

  return success();
}

static LogicalResult NAryInferReturnTypes(
    const ValueShapeRange &operands,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(operands, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
  } else {
    inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  }
  return success();
}

#define NARY_SHAPE_INFER(OP)                                                   \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::llvm::Optional<Location> location,               \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      RegionRange regions,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return NAryInferReturnTypes(operands, inferredReturnShapes);               \
  }

NARY_SHAPE_INFER(tosa::AbsOp)
NARY_SHAPE_INFER(tosa::AddOp)
NARY_SHAPE_INFER(tosa::ArithmeticRightShiftOp)
NARY_SHAPE_INFER(tosa::BitwiseAndOp)
NARY_SHAPE_INFER(tosa::BitwiseOrOp)
NARY_SHAPE_INFER(tosa::BitwiseXorOp)
NARY_SHAPE_INFER(tosa::BitwiseNotOp)
NARY_SHAPE_INFER(tosa::CastOp)
NARY_SHAPE_INFER(tosa::CeilOp)
NARY_SHAPE_INFER(tosa::ClampOp)
NARY_SHAPE_INFER(tosa::ClzOp)
NARY_SHAPE_INFER(tosa::DivOp)
NARY_SHAPE_INFER(tosa::EqualOp)
NARY_SHAPE_INFER(tosa::ExpOp)
NARY_SHAPE_INFER(tosa::FloorOp)
NARY_SHAPE_INFER(tosa::GreaterEqualOp)
NARY_SHAPE_INFER(tosa::GreaterOp)
NARY_SHAPE_INFER(tosa::IdentityOp)
NARY_SHAPE_INFER(tosa::LogOp)
NARY_SHAPE_INFER(tosa::LogicalAndOp)
NARY_SHAPE_INFER(tosa::LogicalLeftShiftOp)
NARY_SHAPE_INFER(tosa::LogicalNotOp)
NARY_SHAPE_INFER(tosa::LogicalOrOp)
NARY_SHAPE_INFER(tosa::LogicalRightShiftOp)
NARY_SHAPE_INFER(tosa::LogicalXorOp)
NARY_SHAPE_INFER(tosa::MaximumOp)
NARY_SHAPE_INFER(tosa::MinimumOp)
NARY_SHAPE_INFER(tosa::MulOp)
NARY_SHAPE_INFER(tosa::NegateOp)
NARY_SHAPE_INFER(tosa::PowOp)
NARY_SHAPE_INFER(tosa::ReciprocalOp)
NARY_SHAPE_INFER(tosa::ReluNOp)
NARY_SHAPE_INFER(tosa::RescaleOp)
NARY_SHAPE_INFER(tosa::ReverseOp)
NARY_SHAPE_INFER(tosa::RsqrtOp)
NARY_SHAPE_INFER(tosa::SelectOp)
NARY_SHAPE_INFER(tosa::SubOp)
NARY_SHAPE_INFER(tosa::TanhOp)
NARY_SHAPE_INFER(tosa::SigmoidOp)
#undef PRED_SHAPE_INFER

static LogicalResult poolingInferReturnTypes(
    const ValueShapeRange &operands, DictionaryAttr attributes,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape = operands.getShape(0);
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(4, -1);

  // We only know the rank if the input type is unranked.
  if (!inputShape) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Batch and number of channels are identical for pooling layer.
  outputShape[0] = inputShape.getDimSize(0);
  outputShape[3] = inputShape.getDimSize(3);

  int32_t height = inputShape.getDimSize(1);
  int32_t width = inputShape.getDimSize(2);

  llvm::SmallVector<int64_t> kernel;
  llvm::SmallVector<int64_t> stride;
  llvm::SmallVector<int64_t> pad;

  getI64Values(attributes.get("kernel").cast<ArrayAttr>(), kernel);
  getI64Values(attributes.get("stride").cast<ArrayAttr>(), stride);
  getI64Values(attributes.get("pad").cast<ArrayAttr>(), pad);

  if (height != -1) {
    int32_t padded = height + pad[0] + pad[1] - kernel[0];
    outputShape[1] = padded / stride[0] + 1;
  }

  if (width != -1) {
    int32_t padded = width + pad[2] + pad[3] - kernel[1];
    outputShape[2] = padded / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamicSize);
  Conv2DOp::Adaptor adaptor(operands.getValues(), attributes);

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.

  ShapeAdaptor inputShape = operands.getShape(adaptor.input());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.weight());
  if (weightShape.hasRank()) {
    outputShape[3] = weightShape.getDimSize(0);
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.bias());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(adaptor.dilation(), dilation);
  getI64Values(adaptor.pad(), padding);
  getI64Values(adaptor.stride(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + padding[0] + padding[1];
    int32_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + padding[2] + padding[3];
    int32_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv3DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(5, ShapedType::kDynamicSize);
  Conv2DOp::Adaptor adaptor(operands.getValues(), attributes);

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t inputDepth = ShapedType::kDynamicSize;

  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;
  int32_t weightDepth = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape = operands.getShape(adaptor.input());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
    inputDepth = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.weight());
  if (weightShape.hasRank()) {
    outputShape[4] = weightShape.getDimSize(0);
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
    weightDepth = weightShape.getDimSize(3);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.bias());
  if (biasShape.hasRank()) {
    outputShape[4] =
        (outputShape[4] == -1) ? biasShape.getDimSize(0) : outputShape[4];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(adaptor.dilation(), dilation);
  getI64Values(adaptor.pad(), padding);
  getI64Values(adaptor.stride(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + padding[0] + padding[1];
    int32_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + padding[2] + padding[3];
    int32_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  if (!ShapedType::isDynamic(inputDepth) &&
      !ShapedType::isDynamic(weightDepth)) {
    int32_t inputSize = inputDepth + padding[4] + padding[5];
    int32_t filterSize = (weightDepth - 1) * dilation[2] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[3] = (unstridedResult - 1) / stride[2] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult AvgPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return poolingInferReturnTypes(operands, attributes, inferredReturnShapes);
}

LogicalResult MaxPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  return poolingInferReturnTypes(operands, attributes, inferredReturnShapes);
}

LogicalResult DepthwiseConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamicSize);
  DepthwiseConv2DOp::Adaptor adaptor(operands.getValues(), attributes);

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t inputChannels = ShapedType::kDynamicSize;

  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;
  int32_t depthChannels = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape = operands.getShape(adaptor.input());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
    inputChannels = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.weight());
  if (weightShape.hasRank()) {
    weightHeight = weightShape.getDimSize(0);
    weightWidth = weightShape.getDimSize(1);
    inputChannels = ShapedType::isDynamic(inputChannels)
                        ? weightShape.getDimSize(2)
                        : inputChannels;
    depthChannels = weightShape.getDimSize(3);
  }

  // If both inputChannels and depthChannels are available we can determine
  // the output channels.
  if (!ShapedType::isDynamic(inputChannels) &&
      !ShapedType::isDynamic(depthChannels)) {
    outputShape[3] = inputChannels * depthChannels;
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.bias());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(adaptor.dilation(), dilation);
  getI64Values(adaptor.pad(), padding);
  getI64Values(adaptor.stride(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t inputSize = inputHeight + padding[0] + padding[1];
    int32_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t inputSize = inputWidth + padding[2] + padding[3];
    int32_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult TransposeConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  TransposeConv2DOp::Adaptor adaptor(operands.getValues(), attributes);
  llvm::SmallVector<int64_t> outputShape;
  getI64Values(adaptor.out_shape(), outputShape);

  int32_t inputWidth = ShapedType::kDynamicSize;
  int32_t inputHeight = ShapedType::kDynamicSize;
  int32_t weightWidth = ShapedType::kDynamicSize;
  int32_t weightHeight = ShapedType::kDynamicSize;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape = operands.getShape(adaptor.input());
  if (inputShape.hasRank()) {
    outputShape[0] = ShapedType::isDynamic(outputShape[0])
                         ? inputShape.getDimSize(0)
                         : outputShape[0];
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape = operands.getShape(adaptor.filter());
  if (weightShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? weightShape.getDimSize(0)
                         : outputShape[3];
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape = operands.getShape(adaptor.input());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::SmallVector<int64_t> dilation;
  llvm::SmallVector<int64_t> padding;
  llvm::SmallVector<int64_t> stride;

  getI64Values(adaptor.dilation(), dilation);
  getI64Values(adaptor.out_pad(), padding);
  getI64Values(adaptor.stride(), stride);

  if (!ShapedType::isDynamic(inputHeight) &&
      !ShapedType::isDynamic(weightHeight)) {
    int32_t dilated = (weightHeight - 1) * dilation[0] + 1;
    int32_t calculateSize =
        (inputHeight - 1) * stride[0] - padding[0] + dilated;
    outputShape[1] = outputShape[1] == -1 ? calculateSize : outputShape[1];
  }

  if (!ShapedType::isDynamic(inputWidth) &&
      !ShapedType::isDynamic(weightWidth)) {
    int32_t dilated = (weightWidth - 1) * dilation[1] + 1;
    int32_t calculateSize = (inputWidth - 1) * stride[1] - padding[1] + dilated;
    outputShape[2] = outputShape[2] == -1 ? calculateSize : outputShape[2];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult IfOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (Region *region : regions) {
    for (auto &block : *region)
      if (auto returnOp = dyn_cast<tosa::YieldOp>(block.getTerminator()))
        yieldOps.push_back(returnOp);
  }

  if (yieldOps.empty())
    return failure();

  // Get the initial type information for the yield op.
  llvm::SmallVector<ValueKnowledge> resultKnowledge;
  resultKnowledge.reserve(yieldOps.front().getNumOperands());
  for (auto operand : yieldOps.front().getOperands()) {
    resultKnowledge.push_back(
        ValueKnowledge::getKnowledgeFromType(operand.getType()));
  }

  for (auto yieldOp : yieldOps) {
    if (resultKnowledge.size() != yieldOp.getNumOperands())
      return failure();

    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
      int32_t index = it.index();
      auto meet = ValueKnowledge::meet(
          resultKnowledge[index],
          ValueKnowledge::getKnowledgeFromType(it.value().getType()));
      if (!meet)
        continue;
      resultKnowledge[index] = meet;
    }
  }

  for (const ValueKnowledge &result : resultKnowledge) {
    inferredReturnShapes.push_back(result.getShapedTypeComponents());
  }

  return success();
}

LogicalResult WhileOp::inferReturnTypeComponents(
    MLIRContext *context, ::llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (auto &block : *regions[1])
    if (auto returnOp = dyn_cast<tosa::YieldOp>(block.getTerminator()))
      yieldOps.push_back(returnOp);

  // TOSA's while must have a tosa.yield as its terminator. If not found this
  // tosa.while is invalid.
  if (yieldOps.empty())
    return failure();

  // Get the initial type information from the operand types.
  llvm::SmallVector<ValueKnowledge> resultKnowledge;
  resultKnowledge.reserve(yieldOps.front().getNumOperands());
  for (auto operand : yieldOps.front().getOperands()) {
    resultKnowledge.push_back(
        ValueKnowledge::getKnowledgeFromType(operand.getType()));
  }

  for (auto yieldOp : yieldOps) {
    if (resultKnowledge.size() != yieldOp.getNumOperands())
      return failure();

    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
      int32_t index = it.index();
      if (auto meet = ValueKnowledge::meet(
              resultKnowledge[index],
              ValueKnowledge::getKnowledgeFromType(it.value().getType()))) {
        resultKnowledge[index] = meet;
      };
    }
  }

  for (const ValueKnowledge &result : resultKnowledge) {
    inferredReturnShapes.push_back(result.getShapedTypeComponents());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
