//===- BufferPlacement.h - Buffer Assignment Utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines buffer assignment helper methods to compute correct
// and valid positions for placing Alloc and Dealloc operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_BUFFERPLACEMENT_H
#define MLIR_TRANSFORMS_BUFFERPLACEMENT_H

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

/// Prepares a buffer placement phase. It can place (user-defined) alloc
/// nodes. This simplifies the integration of the actual buffer-placement
/// pass. Sample usage:
///   BufferAssignmentPlacer baHelper(regionOp);
///   -> determine alloc positions
///   auto allocPosition = baHelper.computeAllocPosition(value);
///   -> place alloc
///   allocBuilder.setInsertionPoint(positions.getAllocPosition());
///   <create alloc>
/// Note: this class is intended to be used during legalization. In order
/// to move alloc and dealloc nodes into the right places you can use the
/// createBufferPlacementPass() function.
class BufferAssignmentPlacer {
public:
  /// Creates a new assignment builder.
  explicit BufferAssignmentPlacer(Operation *op);

  /// Returns the operation this analysis was constructed from.
  Operation *getOperation() const { return operation; }

  /// Computes the actual position to place allocs for the given result.
  OpBuilder::InsertPoint computeAllocPosition(OpResult result);

private:
  /// The operation this analysis was constructed from.
  Operation *operation;
};

/// Helper conversion pattern that encapsulates a BufferAssignmentPlacer
/// instance. Sample usage:
/// class CustomConversionPattern : public
///     BufferAssignmentOpConversionPattern<MyOpT>
/// {
///   ... matchAndRewrite(...) {
///     -> Access stored BufferAssignmentPlacer
///     bufferAssignment->computeAllocPosition(resultOp);
///   }
/// };
template <typename SourceOp>
class BufferAssignmentOpConversionPattern
    : public OpConversionPattern<SourceOp> {
public:
  explicit BufferAssignmentOpConversionPattern(
      MLIRContext *context, BufferAssignmentPlacer *bufferAssignment = nullptr,
      TypeConverter *converter = nullptr, PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit),
        bufferAssignment(bufferAssignment), converter(converter) {}

protected:
  BufferAssignmentPlacer *bufferAssignment;
  TypeConverter *converter;
};

/// A helper type converter class for using inside Buffer Assignment operation
/// conversion patterns. The default constructor keeps all the types intact
/// except for the ranked-tensor types which is converted to memref types.
class BufferAssignmentTypeConverter : public TypeConverter {
public:
  BufferAssignmentTypeConverter();

  /// A helper function to check if `type` has been converted from non-memref
  /// type to memref.
  static bool isConvertedMemref(Type type, Type before);
};

namespace detail {

/// Converts the signature of the function based on whether the function is
/// allowed to return memref typed results or not using
/// `allowMemrefFunctionResults` parameter. If this option is false, then it
/// adds an extra function argument as an output buffer for each function result
/// which is going to be a memref type only after type conversion. The
/// other function result types remain unchanged. If
/// `allowMemrefFunctionResults` is true, the types are converted in place.
/// Any changes in function signature need to be applied
/// to return and caller operations. `BufferAssignmentReturnOpConverter` and
/// `BufferAssignmentCallOpConverter` are two helper function that match the
/// return and caller operation with the new function signature. Furthermore,
/// `BufferAssignmentTypeConverter` is a helper `TypeConverter` for converting
/// tensor typed values to memref typed ones.
template <bool allowMemrefFunctionResults>
class BufferAssignmentFuncOpConverter
    : public BufferAssignmentOpConversionPattern<FuncOp> {
public:
  using BufferAssignmentOpConversionPattern<
      FuncOp>::BufferAssignmentOpConversionPattern;

  /// Performs the actual signature rewriting step.
  LogicalResult
  matchAndRewrite(mlir::FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!converter)
      return funcOp.emitError("The type converter has not been defined for "
                              "BufferAssignmentFuncOpConverter");
    auto funcType = funcOp.getType();

    // Convert function arguments using the provided TypeConverter.
    TypeConverter::SignatureConversion conversion(funcType.getNumInputs());
    for (auto argType : llvm::enumerate(funcType.getInputs()))
      conversion.addInputs(argType.index(),
                           converter->convertType(argType.value()));

    // If allowMemrefFunctionResults is false and a function result type is not
    // a memref but it would be a memref after type conversion, a new argument
    // should be appended to the function arguments list for this result.
    // Otherwise, it remains unchanged as a function result.
    SmallVector<Type, 2> newResultTypes;
    newResultTypes.reserve(funcOp.getNumResults());
    for (Type resType : funcType.getResults()) {
      Type convertedType = converter->convertType(resType);
      if (!allowMemrefFunctionResults &&
          BufferAssignmentTypeConverter::isConvertedMemref(convertedType,
                                                           resType))
        conversion.addInputs(convertedType);
      else
        newResultTypes.push_back(convertedType);
    }
    if (failed(rewriter.convertRegionTypes(&funcOp.getBody(), *converter,
                                           &conversion)))
      return failure();

    // Update the signature of the function.
    rewriter.updateRootInPlace(funcOp, [&] {
      funcOp.setType(rewriter.getFunctionType(conversion.getConvertedTypes(),
                                              newResultTypes));
    });
    return success();
  }
};

/// Rewrites the `ReturnOp` to conform with the changed function signature.
/// if allowMemrefFunctionResults is false, operands that correspond to return
/// values and have been rewritten from illegal typed results to memref
/// arguments are dropped. In their place, a corresponding copy operation from
/// the operand to the output function argument is inserted. Otherwise, the
/// memref typed operands are returned.
/// Note: If this pattern rewriter is used with BufferAssignmentFuncOpConverter,
/// allowMemrefFunctionResults must be set/unset for both.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy, bool allowMemrefFunctionResults>
class BufferAssignmentReturnOpConverter
    : public BufferAssignmentOpConversionPattern<ReturnOpSourceTy> {
public:
  using BufferAssignmentOpConversionPattern<
      ReturnOpSourceTy>::BufferAssignmentOpConversionPattern;

  /// Performs the actual return-op conversion step.
  LogicalResult
  matchAndRewrite(ReturnOpSourceTy returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // If the memref typed results can be returned as function results, the new
    // `ReturnOp` should only return the type converted operands.
    if (allowMemrefFunctionResults) {
      rewriter.replaceOpWithNewOp<ReturnOpTargetTy>(returnOp, operands);
      return success();
    }

    // Split the operands by their kinds whether they are converted memref or
    // not.
    SmallVector<Value, 2> needCopyOperands, newOperands;
    unsigned operandsSize = operands.size();
    needCopyOperands.reserve(operandsSize);
    newOperands.reserve(operandsSize);
    for (auto operand : llvm::enumerate(operands))
      if (BufferAssignmentTypeConverter::isConvertedMemref(
              operand.value().getType(),
              returnOp.getOperand(operand.index()).getType()))
        needCopyOperands.push_back(operand.value());
      else
        newOperands.push_back(operand.value());

    Block &entryBlock = returnOp.getParentRegion()->front();
    unsigned numFuncArgs = entryBlock.getNumArguments();

    // Find the index of the first destination buffer.
    assert(needCopyOperands.size() <= numFuncArgs &&
           "The number of operands of return operation is more than the "
           "number of function arguments.");
    unsigned destArgNum = numFuncArgs - needCopyOperands.size();
    rewriter.setInsertionPoint(returnOp);
    for (Value operand : needCopyOperands) {
      // Insert a `CopyOp` for each converted memref-type operand.
      rewriter.create<CopyOpTy>(returnOp.getLoc(), operand,
                                entryBlock.getArgument(destArgNum));
      ++destArgNum;
    }

    // Insert the new target Return operation.
    rewriter.replaceOpWithNewOp<ReturnOpTargetTy>(returnOp, newOperands);
    return success();
  }
};

/// Rewrites the `CallOp` to match its operands and results with the signature
/// of the callee after rewriting the callee with
/// BufferAssignmentFuncOpConverter. If allowMemrefFunctionResults is false, a
/// buffer is allocated as an output buffer only for each memref typed result
/// that has been rewritten. The new allocated buffer is passed through the
/// operands list of the new `CallOp`.
/// Note: If this pattern rewriter is used with BufferAssignmentFuncOpConverter,
/// allowMemrefFunctionResults must be set/unset for both.
template <bool allowMemrefFunctionResults>
class BufferAssignmentCallOpConverter
    : public BufferAssignmentOpConversionPattern<CallOp> {
public:
  using BufferAssignmentOpConversionPattern<
      CallOp>::BufferAssignmentOpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp callOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    if (!converter)
      return callOp.emitError("The type converter has not been defined for "
                              "BufferAssignmentCallOpConverter");
    Location loc = callOp.getLoc();

    // If the memref typed results can be returned as function results, there is
    // no need to create output buffers. It is only required to convert the type
    // of operands and results in place for creating the new `CallOp`.
    if (allowMemrefFunctionResults) {
      SmallVector<Type, 2> resultTypes;
      resultTypes.reserve(callOp.getNumResults());
      for (Type type : callOp.getResultTypes())
        resultTypes.push_back(converter->convertType(type));
      rewriter.replaceOpWithNewOp<CallOp>(callOp, callOp.getCallee(),
                                          resultTypes, operands);
      return success();
    }

    SmallVector<Value, 2> newOperands, replacingValues;
    SmallVector<Type, 2> newResultTypes;
    unsigned numResults = callOp.getNumResults();
    newOperands.reserve(numResults + operands.size());
    newOperands.append(operands.begin(), operands.end());
    newResultTypes.reserve(numResults);
    replacingValues.reserve(numResults);

    // For each memref result of `CallOp` which has not been a memref before
    // the type conversion, a new buffer is allocated and passed to the operands
    // list of the new `CallOp`. Otherwise, it remains as a caller result.
    for (Value result : callOp.getResults()) {
      Type currType = result.getType();
      Type newType = converter->convertType(result.getType());
      if (BufferAssignmentTypeConverter::isConvertedMemref(newType, currType)) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.restoreInsertionPoint(bufferAssignment->computeAllocPosition(
            result.dyn_cast<OpResult>()));
        Value alloc =
            rewriter.create<AllocOp>(loc, newType.dyn_cast<MemRefType>());
        newOperands.push_back(alloc);
        replacingValues.push_back(alloc);
      } else {
        newResultTypes.push_back(currType);

        // No replacing is required.
        replacingValues.push_back(nullptr);
      }
    }

    // Creating the new `CallOp`.
    rewriter.create<CallOp>(loc, callOp.getCallee(), newResultTypes,
                            newOperands);

    // Replacing the results of the old `CallOp`.
    rewriter.replaceOp(callOp, replacingValues);
    return success();
  }
};
} // end namespace detail

/// Populates `patterns` with the conversion patterns of buffer
/// assignment.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy, bool allowMemrefFunctionResults>
static void populateWithBufferAssignmentOpConversionPatterns(
    MLIRContext *context, BufferAssignmentPlacer *placer,
    TypeConverter *converter, OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<
    detail::BufferAssignmentCallOpConverter<allowMemrefFunctionResults>,
    detail::BufferAssignmentFuncOpConverter<allowMemrefFunctionResults>,
    detail::BufferAssignmentReturnOpConverter
      <ReturnOpSourceTy, ReturnOpTargetTy, CopyOpTy, allowMemrefFunctionResults>
  >(context, placer, converter);
  // clang-format on
}
} // end namespace mlir

#endif // MLIR_TRANSFORMS_BUFFERPLACEMENT_H
