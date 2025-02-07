//===- ConversionUtils.h - Helper functions for tosa conversion -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions for TOSA lowering
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_TOSA_UTILS_COVERSION_UTILS_H_
#define DIALECT_TOSA_UTILS_COVERSION_UTILS_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>

namespace mlir {
namespace tosa {

// Creates a SmallVector of Stringrefs for N parallel loops
SmallVector<utils::IteratorType>
getNParallelLoopsAttrs(unsigned nParallelLoops);

// Takes a vector of values and condenses them to a vector with no gaps.
SmallVector<Value> condenseValues(const SmallVector<Value> &values);

// Takes the parameters for a clamp and turns it into a series of ops for float
// inputs.
Value clampFloatHelper(Location loc, Value arg, Value min, Value max,
                       OpBuilder &rewriter);

// Takes the parameters for a clamp and turns it into a series of ops for
// integer inputs.
Value clampIntHelper(Location loc, Value arg, Value min, Value max,
                     OpBuilder &rewriter, bool isUnsigned);

// Determines whether the integer value falls witin the range of integer type.
bool validIntegerRange(IntegerType ty, int64_t value);

// Checks for a dynamic batch dim in any of the passed parameters of an op.
// The batch dimention must be #0 and the rest of the dimensions must be static.
template <typename Op>
std::optional<SmallVector<Value>>
checkHasDynamicBatchDims(PatternRewriter &rewriter, Op op,
                         ArrayRef<Value> params) {
  SmallVector<ShapedType> dynTypes;
  SmallVector<Value> dynamicDims;
  for (const Value &param : params) {
    auto paramTy = cast<ShapedType>(param.getType());
    if (!paramTy.hasStaticShape())
      dynTypes.push_back(paramTy);
  }

  if (dynTypes.empty())
    return dynamicDims;

  for (const ShapedType &dynTy : dynTypes) {
    if (llvm::any_of(dynTy.getShape().drop_front(), ShapedType::isDynamic)) {
      (void)rewriter.notifyMatchFailure(
          op, "input can only be dynamic for batch size");
      return std::nullopt;
    }
  }

  dynamicDims.push_back(
      rewriter.create<tensor::DimOp>(op->getLoc(), params[0], 0));
  return dynamicDims;
}

/// Common code to create the reshape op where necessary to make the rank of two
/// values equal. input1 and input2 will be updated when the rank has
/// changed. The caller is expected to use these to rewrite the original
/// operator with the RESHAPE now in the graph.
LogicalResult EqualizeRanks(PatternRewriter &rewriter, Location loc,
                            Value &input1, Value &input2);

LogicalResult EqualizeRanks(ImplicitLocOpBuilder &builder, Value &input1,
                            Value &input2);

namespace {

// Creates a TOSA operation and performs shape inference on the individual
// op. This allows shape inference when lowering down to TOSA.
template <typename TosaOp, typename... Args>
TosaOp createOpAndInferShape(ImplicitLocOpBuilder &builder, Type resultTy,
                             Args &&...args) {
  auto op = builder.create<TosaOp>(resultTy, args...);

  InferShapedTypeOpInterface shapeInterface =
      dyn_cast<InferShapedTypeOpInterface>(op.getOperation());
  if (!shapeInterface)
    return op;

  SmallVector<ShapedTypeComponents> returnedShapes;
  if (shapeInterface
          .inferReturnTypeComponents(op.getContext(), builder.getLoc(),
                                     op->getOperands(), op->getAttrDictionary(),
                                     op->getPropertiesStorage(),
                                     op->getRegions(), returnedShapes)
          .failed())
    return op;

  // We need to use the element type of the existing result type to generate
  // the new result shaped type. This is because rescale can include a cast to
  // different bit-width types and does not have a TypeAttr to define the
  // target type.
  auto result = op->getResult(0);
  auto predictedShape = returnedShapes[0];
  auto currentKnowledge = ValueKnowledge::getKnowledgeFromType(resultTy);

  // Compute the knowledge based on the inferred type.
  auto inferredKnowledge = ValueKnowledge::getPessimisticValueState();
  inferredKnowledge.dtype = mlir::cast<ShapedType>(resultTy).getElementType();
  inferredKnowledge.hasRank = predictedShape.hasRank();
  if (predictedShape.hasRank()) {
    for (auto dim : predictedShape.getDims()) {
      inferredKnowledge.sizes.push_back(dim);
    }
  }

  // Compute the new type based on the joined version.
  auto newKnowledge = ValueKnowledge::join(currentKnowledge, inferredKnowledge);
  Type newTy =
      newKnowledge.hasRank
          ? Type{mlir::RankedTensorType::get(llvm::ArrayRef(newKnowledge.sizes),
                                             newKnowledge.dtype)}
          : Type{mlir::UnrankedTensorType::get(newKnowledge.dtype)};
  result.setType(newTy);
  return op;
}

} // namespace

// Creates a TOSA operation by:
//   - first equalize ranks for ops with SameOperandsAndResultRank trait
//   - create operator
//   - performs shape inference on this operator
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInferShape(ImplicitLocOpBuilder &builder, Type resultTy,
                             Args &&...args) {
  if (TosaOp::template hasTrait<OpTrait::SameOperandsAndResultRank>()) {
    // op requires same ranks for tensor operands
    if constexpr (sizeof...(Args) == 2) {
      auto argX = std::get<0>(std::tie(args...));
      auto argY = std::get<1>(std::tie(args...));
      using ArgX = decltype(argX);
      using ArgY = decltype(argY);
      if constexpr (std::is_same_v<ArgX, Value> &&
                    std::is_same_v<ArgY, Value>) {
        Value x = std::get<0>(std::tie(args...));
        Value y = std::get<1>(std::tie(args...));
        if (EqualizeRanks(builder, x, y).failed()) {
          // incompatible broadcast shapes, no reshape is inserted
          // ResultsBroadcastableShape verify will handle this
        }
        return createOpAndInferShape<TosaOp>(builder, resultTy, x, y);
      }
    }
    if constexpr (sizeof...(Args) == 3) {
      auto argX = std::get<0>(std::tie(args...));
      auto argY = std::get<1>(std::tie(args...));
      auto argZ = std::get<2>(std::tie(args...));
      using ArgX = decltype(argX);
      using ArgY = decltype(argY);
      using ArgZ = decltype(argZ);
      if constexpr (std::is_same_v<ArgX, Value> &&
                    std::is_same_v<ArgY, Value> && std::is_same_v<ArgZ, bool>) {
        // special case for ArithmeticRightShiftOp
        Value x = std::get<0>(std::tie(args...));
        Value y = std::get<1>(std::tie(args...));
        bool round = std::get<2>(std::tie(args...));
        if (EqualizeRanks(builder, x, y).failed()) {
          // incompatible broadcast shapes, no reshape is inserted
          // ResultsBroadcastableShape verify will handle this
        }
        return createOpAndInferShape<TosaOp>(builder, resultTy, x, y, round);
      }
      if constexpr (std::is_same_v<ArgX, Value> &&
                    std::is_same_v<ArgY, Value> &&
                    std::is_same_v<ArgZ, Value>) {
        // special case for Select
        Value x = std::get<0>(std::tie(args...));
        Value y = std::get<1>(std::tie(args...));
        Value z = std::get<2>(std::tie(args...));

        if (EqualizeRanks(builder, x, y).failed() ||
            EqualizeRanks(builder, x, z).failed() ||
            EqualizeRanks(builder, y, z).failed()) {
          // incompatible broadcast shapes, no reshape is inserted
          // ResultsBroadcastableShape verify will handle this
        }

        return createOpAndInferShape<TosaOp>(builder, resultTy, x, y, z);
      }
    }
  }

  return createOpAndInferShape<TosaOp>(builder, resultTy, args...);
}

// Creates a TOSA operation by:
//   - first equalize ranks for ops with SameOperandsAndResultRank trait
//   - create operator
//   - performs shape inference on this operator
template <typename TosaOp, typename... Args>
TosaOp CreateOpAndInferShape(PatternRewriter &rewriter, Location loc,
                             Type resultTy, Args &&...args) {
  ImplicitLocOpBuilder builder(loc, rewriter);
  return CreateOpAndInferShape<TosaOp>(builder, resultTy, args...);
}

// Apply an int32_t permutation to some input, that should be of the same
// size as perms. Perms should contain some permutation of 0 - perms.size() - 1.
template <typename T>
SmallVector<T> applyTOSAPermutation(ArrayRef<T> input,
                                    ArrayRef<int32_t> perms) {
  SmallVector<T> permuted;
  size_t N = input.size();
  permuted.resize_for_overwrite(N);
  for (size_t i = 0; i < N; i++)
    permuted[i] = input[perms[i]];
  return permuted;
}

// Computes shape value using tosa const_shape op.
Value getTosaConstShape(ImplicitLocOpBuilder &builder,
                        llvm::ArrayRef<int64_t> shape);
Value getTosaConstShape(PatternRewriter &rewriter, Location loc,
                        llvm::ArrayRef<int64_t> shape);

SmallVector<int64_t> convertFromMlirShape(ArrayRef<int64_t> shape);

bool getConstShapeValue(Operation *op,
                        llvm::SmallVector<int64_t> &result_shape);

} // namespace tosa
} // namespace mlir

#endif // DIALECT_TOSA_UTILS_COVERSION_UTILS_H_
