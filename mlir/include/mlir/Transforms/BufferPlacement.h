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

/// Converts the signature of the function using the type converter. It adds an
/// extra argument for each function result type which is going to be a memref
/// type after type conversion. The other function result types remain
/// unchanged. `BufferAssignmentTypeConverter` is a helper `TypeConverter` for
/// this purpose.
class FunctionAndBlockSignatureConverter
    : public BufferAssignmentOpConversionPattern<FuncOp> {
public:
  using BufferAssignmentOpConversionPattern<
      FuncOp>::BufferAssignmentOpConversionPattern;

  /// Performs the actual signature rewriting step.
  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final;
};

/// Rewrites the `ReturnOp` to conform with the changed function signature.
/// Operands that correspond to return values that have been rewritten from
/// tensor results to memref arguments are dropped. In their place, a
/// corresponding copy operation from the operand to the new function argument
/// is inserted.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy>
class BufferAssignmentReturnOpConverter
    : public BufferAssignmentOpConversionPattern<ReturnOpSourceTy> {
public:
  using BufferAssignmentOpConversionPattern<
      ReturnOpSourceTy>::BufferAssignmentOpConversionPattern;

  /// Performs the actual return-op conversion step.
  LogicalResult
  matchAndRewrite(ReturnOpSourceTy returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
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
} // end namespace mlir

#endif // MLIR_TRANSFORMS_BUFFERPLACEMENT_H
