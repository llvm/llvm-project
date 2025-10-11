//===- TosaFolders.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Fold TOSA operations
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <numeric>

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::tosa;

namespace {

/// Apply the given transformation \p toApply to every element of the tensor to
/// be transformed \p toTransform.
///
/// Elements of \p toTransform are extracted as \p SrcValueType.
///
/// \returns A tensor with the same size as \p toTransform, containing
/// \p TargetValueType values of type \p TargetType.
template <class SrcValType, class TargetValType, class TargetType>
DenseElementsAttr applyElementWise(
    const DenseElementsAttr &toTransform,
    const std::function<TargetValType(const SrcValType &)> &toApply,
    TargetType targetType) {
  SmallVector<TargetValType> transformedValues;
  // We already know the amount of values we will insert, reserve space for
  // all of them to avoid dynamic resizing
  transformedValues.reserve(toTransform.getNumElements());
  for (auto val : toTransform.getValues<SrcValType>()) {
    auto transformedVal = toApply(val);
    transformedValues.push_back(transformedVal);
  }

  // Make sure that the output tensor has the expected output type
  auto inShape = toTransform.getType();
  auto outTy = inShape.cloneWith({}, targetType);

  return DenseElementsAttr::get(outTy, transformedValues);
}

template DenseElementsAttr applyElementWise<APFloat, APFloat, FloatType>(
    const DenseElementsAttr &toTransform,
    const std::function<APFloat(const APFloat &)> &toApply,
    FloatType targetType);

/// Function that checks if the type contained in \p toCheck is float.
LogicalResult notifyIfNotFloat(TypedValue<TensorType> toCheck, TosaOp location,
                               PatternRewriter &rewriter) {
  if (isa<FloatType>(toCheck.getType().getElementType())) {
    return success();
  }
  return rewriter.notifyMatchFailure(location,
                                     "Unexpected input tensor type: the "
                                     "TOSA spec only allows floats");
}

/// Function that checks if \p toCheck is a dense TOSA constant tensor.
LogicalResult notifyIfNoTosaDenseConstantTensor(TypedValue<TensorType> toCheck,
                                                TosaOp location,
                                                PatternRewriter &rewriter) {
  // Check whether the tensor is constant and dense
  // TODO We currently ensure the tensor is dense by using the correct type for
  // the bind_value, however we do not actually need this value. It would be
  // nicer to only have a check here.
  DenseElementsAttr tmp;
  if (!matchPattern(toCheck, m_Constant(&tmp))) {
    return rewriter.notifyMatchFailure(location,
                                       "Non-const or non-dense input tensor");
  }

  // Make sure it actually is a TOSA constant (the match allows for other
  // constants as well)
  if (isa<ConstOp>(toCheck.getDefiningOp())) {
    return success();
  }

  return rewriter.notifyMatchFailure(location,
                                     "The reciprocal can only be folded if "
                                     "it operates on a TOSA constant");
}

/// Function that checks if \p toCheck is a dense TOSA constant float tensor.
LogicalResult notifyIfNotConstantFloatTosaTensor(TypedValue<TensorType> toCheck,
                                                 TosaOp location,
                                                 PatternRewriter &rewriter) {
  auto floatCheck = notifyIfNotFloat(toCheck, location, rewriter);
  if (failed(floatCheck)) {
    return floatCheck;
  }
  return notifyIfNoTosaDenseConstantTensor(toCheck, location, rewriter);
}

/// Heuristic to decide when to replace a unary operation on a constant with the
/// folded value.
/// Folding operations on constants can lead to an increased memory usage
/// whenever the input cannot be replaced but a new constant is inserted. Hence,
/// this will currently only suggest folding when the memory impact is
/// negligible.
/// Takes the \p unaryOp and the constant input \p values.
/// \returns Whether folding should be applied.
bool constantUnaryOpShouldBeFolded(TosaOp unaryOp, DenseElementsAttr values) {
  assert(unaryOp->getNumOperands() == 1);
  auto inputOp = unaryOp->getOperand(0);

  // If the input is a splat, we don't care for the number of users
  if (isa<SplatElementsAttr>(values)) {
    return true;
  }

  // If this is the only use of the tensor it should be replaced as no
  // additional memory is required
  return inputOp.hasOneUse();
}

template <typename RangeType>
DenseElementsAttr transposeType(const RangeType &data, ShapedType inputType,
                                ShapedType outputType,
                                llvm::ArrayRef<int64_t> permValues) {
  using ElementType = std::decay_t<decltype(*std::begin(data))>;

  assert(inputType.getElementType() == outputType.getElementType());

  if (inputType.getNumElements() == 0)
    return DenseElementsAttr::get(outputType, llvm::ArrayRef<ElementType>{});

  auto inputShape = inputType.getShape();

  // The inverted permutation map and strides of the output are used to compute
  // the contribution of a given dimension to the destination linear index in
  // an order-independent way.
  auto outputStrides = computeStrides(outputType.getShape());
  auto invertedPermValues = invertPermutationVector(permValues);

  auto initialValue = *std::begin(data);
  SmallVector<ElementType> outputValues(inputType.getNumElements(),
                                        initialValue);

  for (const auto &it : llvm::enumerate(data)) {
    auto srcLinearIndex = it.index();

    uint64_t dstLinearIndex = 0;
    for (int64_t dim = inputShape.size() - 1; dim >= 0; --dim) {
      // Compute the index into the current dimension of the source vector.
      auto sourceIndexForDim = srcLinearIndex % inputShape[dim];
      srcLinearIndex /= inputShape[dim];

      // Add the contribution of the current dimension to the output using the
      // permutation map.
      dstLinearIndex +=
          outputStrides[invertedPermValues[dim]] * sourceIndexForDim;
    }

    outputValues[dstLinearIndex] = it.value();
  }

  return DenseElementsAttr::get(outputType,
                                llvm::ArrayRef<ElementType>(outputValues));
}

// Try to get the values of a DenseResourceElementsAttr construct
template <typename T>
std::optional<ArrayRef<T>> tryGetDenseResourceValues(ElementsAttr attr) {
  if (auto denseResource = dyn_cast<DenseResourceElementsAttr>(attr)) {
    // Check that the resource memory blob exists
    AsmResourceBlob *blob = denseResource.getRawHandle().getBlob();
    if (!blob)
      return std::nullopt;

    // Check that the data are in a valid form
    bool isSplat = false;
    if (!DenseElementsAttr::isValidRawBuffer(attr.getShapedType(),
                                             blob->getData(), isSplat)) {
      return std::nullopt;
    }

    return blob->template getDataAs<T>();
  }

  return std::nullopt;
}

// A type specialized transposition of an ElementsAttr.
// This implementation tries to operate on the underlying data in its raw
// representation when possible to avoid allocating a large number of Attribute
// objects.
DenseElementsAttr transpose(ElementsAttr attr, ShapedType inputType,
                            ShapedType outputType,
                            llvm::ArrayRef<int64_t> permValues) {
  // Handle generic ElementsAttr
  if (auto data = attr.tryGetValues<bool>())
    return transposeType(*data, inputType, outputType, permValues);

  if (auto data = attr.tryGetValues<int8_t>())
    return transposeType(*data, inputType, outputType, permValues);

  if (auto data = attr.tryGetValues<int16_t>())
    return transposeType(*data, inputType, outputType, permValues);

  if (auto data = attr.tryGetValues<int32_t>())
    return transposeType(*data, inputType, outputType, permValues);

  if (auto data = attr.tryGetValues<int64_t>())
    return transposeType(*data, inputType, outputType, permValues);

  if (auto data = attr.tryGetValues<float>())
    return transposeType(*data, inputType, outputType, permValues);

  if (auto data = attr.tryGetValues<APFloat>())
    return transposeType(*data, inputType, outputType, permValues);

  // Handle DenseResourceElementsAttr
  if (isa<DenseResourceElementsAttr>(attr)) {
    auto elementTy = attr.getElementType();

    if (auto data = tryGetDenseResourceValues<bool>(attr);
        data && elementTy.isInteger(1))
      return transposeType(*data, inputType, outputType, permValues);

    if (auto data = tryGetDenseResourceValues<int8_t>(attr);
        data && elementTy.isInteger(8))
      return transposeType(*data, inputType, outputType, permValues);

    if (auto data = tryGetDenseResourceValues<int16_t>(attr);
        data && elementTy.isInteger(16))
      return transposeType(*data, inputType, outputType, permValues);

    if (auto data = tryGetDenseResourceValues<int32_t>(attr);
        data && elementTy.isInteger(32))
      return transposeType(*data, inputType, outputType, permValues);

    if (auto data = tryGetDenseResourceValues<int64_t>(attr);
        data && elementTy.isInteger(64))
      return transposeType(*data, inputType, outputType, permValues);

    if (auto data = tryGetDenseResourceValues<float>(attr);
        data && elementTy.isF32())
      return transposeType(*data, inputType, outputType, permValues);
  }

  return nullptr;
}

struct TosaFoldConstantTranspose : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tosa::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto outputType = cast<ShapedType>(op.getType());
    // TOSA supports quantized types.
    if (!outputType.getElementType().isIntOrIndexOrFloat())
      return failure();

    ElementsAttr inputValues;
    if (!matchPattern(op.getInput1(), m_Constant(&inputValues)))
      return failure();
    // Make sure the input is a constant that has a single user.
    if (!llvm::hasSingleElement(op.getInput1().getDefiningOp()->getUsers()))
      return failure();

    auto permValues = llvm::map_to_vector(
        op.getPerms(), [](const int32_t v) { return static_cast<int64_t>(v); });

    auto inputType = cast<ShapedType>(op.getInput1().getType());

    auto resultAttr = transpose(inputValues, inputType, outputType, permValues);
    if (!resultAttr) {
      return rewriter.notifyMatchFailure(
          op, "unsupported attribute or element type");
    }

    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, outputType, resultAttr);
    return success();
  }
};

struct TosaFoldConstantReciprocal : public OpRewritePattern<ReciprocalOp> {

  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReciprocalOp recip,
                                PatternRewriter &rewriter) const override {
    auto inputTensor = recip.getInput1();

    // Check that we can apply folding
    auto preCondCheck =
        notifyIfNotConstantFloatTosaTensor(inputTensor, recip, rewriter);
    if (failed(preCondCheck)) {
      return preCondCheck;
    }

    // Extract the tensor values
    DenseElementsAttr inputValues;
    matchPattern(inputTensor, m_Constant(&inputValues));

    // Check whether this should be folded.
    if (!constantUnaryOpShouldBeFolded(recip, inputValues)) {
      return rewriter.notifyMatchFailure(
          recip, "Currently, reciprocals will only be folded if the input "
                 "tensor has a single user");
    }

    // Create a new tensor with the updated values
    auto newTensor = applyElementWise<APFloat, APFloat, FloatType>(
        inputValues, &ReciprocalOp::calcOneElement,
        cast<FloatType>(inputValues.getElementType()));

    // Replace the use of the reciprocal with the transformed tensor
    rewriter.replaceOpWithNewOp<ConstOp>(recip, newTensor.getType(), newTensor);
    return success();
  }
};

/// Getting the axes position of the element which is located
/// in the tensor at the counter index

llvm::SmallVector<int64_t>
getPositionFromIndex(int64_t index, llvm::ArrayRef<int64_t> tensorShape) {
  int64_t remaining = index;
  llvm::SmallVector<int64_t> position(tensorShape.size(), 0);
  for (int64_t i = tensorShape.size() - 1; i >= 0; --i) {
    position[i] = remaining % tensorShape[i];
    remaining /= tensorShape[i];
  }
  return position;
}

/// Getting the index of the element which is located at the
/// axes position in the tensor

int64_t getIndexFromPosition(llvm::ArrayRef<int64_t> position,
                             llvm::ArrayRef<int64_t> tensorShape) {
  int64_t index = 0;
  int64_t multiplierTmp = 1;
  for (int64_t i = position.size() - 1; i >= 0; --i) {
    index += position[i] * multiplierTmp;
    multiplierTmp *= tensorShape[i];
  }
  return index;
}

template <typename OperationType>
llvm::APInt calculateReducedValue(const mlir::ElementsAttr &oldTensorAttr,
                                  llvm::ArrayRef<int64_t> oldShape,
                                  int64_t reductionAxis,
                                  int64_t reductionIndex) {

  llvm::SmallVector<int64_t> newShape(oldShape);
  newShape[reductionAxis] = 1;
  /// Let's calculate the position of the index
  llvm::SmallVector<int64_t> position =
      getPositionFromIndex(reductionIndex, newShape);
  auto oldTensor = oldTensorAttr.getValues<llvm::APInt>();
  /// Starting from the first positon along the reduction axis
  position[reductionAxis] = 0;
  int64_t indexAtOldTensor = getIndexFromPosition(position, oldShape);
  llvm::APInt reducedValue = oldTensor[indexAtOldTensor];

  for (int64_t reductionAxisVal = 1; reductionAxisVal < oldShape[reductionAxis];
       ++reductionAxisVal) {

    int64_t stride = llvm::product_of(oldShape.drop_front(reductionAxis + 1));
    int64_t index = indexAtOldTensor + stride * reductionAxisVal;
    reducedValue =
        OperationType::calcOneElement(reducedValue, oldTensor[index]);
  }
  return reducedValue;
}

template <typename OperationType>
struct ReduceConstantOptimization : public OpRewritePattern<OperationType> {

  ReduceConstantOptimization(MLIRContext *context,
                             bool aggressiveReduceConstant)
      : OpRewritePattern<OperationType>(context),
        aggressiveReduceConstant(aggressiveReduceConstant) {}

  using OpRewritePattern<OperationType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OperationType op,
                                PatternRewriter &rewriter) const override {
    Value inputOp = op.getInput();
    auto constOp = inputOp.getDefiningOp<tosa::ConstOp>();

    if (!constOp)
      return rewriter.notifyMatchFailure(
          op, "reduce input must be const operation");

    if (!inputOp.hasOneUse() && !this->aggressiveReduceConstant)
      return rewriter.notifyMatchFailure(
          op, "input operation has more than one user");

    auto resultType = cast<ShapedType>(op.getOutput().getType());

    if (!resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "result type shape is not static");

    auto reductionAxis = op.getAxis();
    const auto denseElementsAttr = constOp.getValues();
    const auto shapedOldElementsValues =
        cast<ShapedType>(denseElementsAttr.getType());

    if (!llvm::isa<IntegerType>(shapedOldElementsValues.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "reduce input currently supported with integer type");

    auto oldShape = shapedOldElementsValues.getShape();
    auto newShape = resultType.getShape();

    int64_t newNumOfElements = llvm::product_of(newShape);
    llvm::SmallVector<APInt> newReducedTensor(newNumOfElements);

    for (int64_t reductionIndex = 0; reductionIndex < newNumOfElements;
         ++reductionIndex) {

      /// Let's reduce all the elements along this reduction axis
      newReducedTensor[reductionIndex] = calculateReducedValue<OperationType>(
          denseElementsAttr, oldShape, reductionAxis, reductionIndex);
    }

    auto rankedTensorType = cast<RankedTensorType>(resultType);
    auto denseAttr =
        mlir::DenseElementsAttr::get(rankedTensorType, newReducedTensor);
    rewriter.replaceOpWithNewOp<tosa::ConstOp>(op, rankedTensorType, denseAttr);
    return success();
  }
  const bool aggressiveReduceConstant;
};

} // namespace

void mlir::tosa::populateTosaConstantReduction(MLIRContext *ctx,
                                               RewritePatternSet &patterns,
                                               bool aggressiveReduceConstant) {
  patterns.add<ReduceConstantOptimization<ReduceAllOp>>(
      ctx, aggressiveReduceConstant);
  patterns.add<ReduceConstantOptimization<ReduceAnyOp>>(
      ctx, aggressiveReduceConstant);
  patterns.add<ReduceConstantOptimization<ReduceMaxOp>>(
      ctx, aggressiveReduceConstant);
  patterns.add<ReduceConstantOptimization<ReduceMinOp>>(
      ctx, aggressiveReduceConstant);
  patterns.add<ReduceConstantOptimization<ReduceProductOp>>(
      ctx, aggressiveReduceConstant);
  patterns.add<ReduceConstantOptimization<ReduceSumOp>>(
      ctx, aggressiveReduceConstant);
}

void mlir::tosa::populateTosaFoldConstantTransposePatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantTranspose>(ctx);
}

void mlir::tosa::populateTosaFoldConstantReciprocalPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<TosaFoldConstantReciprocal>(ctx);
}
