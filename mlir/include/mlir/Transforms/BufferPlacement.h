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

/// Converts the signature of the function using the type converter.
/// It adds an extra argument for each illegally-typed function
/// result to the function arguments. `BufferAssignmentTypeConverter`
/// is a helper `TypeConverter` for this purpose. All the non-shaped types
/// of the input function will be converted to memref.
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

/// Converts the source `ReturnOp` to target `ReturnOp`, removes all
/// the buffer operands from the operands list, and inserts `CopyOp`s
/// for all buffer operands instead.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy>
class NoBufferOperandsReturnOpConverter
    : public BufferAssignmentOpConversionPattern<ReturnOpSourceTy> {
public:
  using BufferAssignmentOpConversionPattern<
      ReturnOpSourceTy>::BufferAssignmentOpConversionPattern;

  /// Performs the actual return-op conversion step.
  LogicalResult
  matchAndRewrite(ReturnOpSourceTy returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Block &entryBlock = returnOp.getParentRegion()->front();
    unsigned numFuncArgs = entryBlock.getNumArguments();
    Location loc = returnOp.getLoc();

    // The target `ReturnOp` should not contain any memref operands.
    SmallVector<Value, 2> newOperands(operands.begin(), operands.end());
    llvm::erase_if(newOperands, [](Value operand) {
      return operand.getType().isa<MemRefType>();
    });

    // Find the index of the first destination buffer.
    unsigned numBufferOperands = operands.size() - newOperands.size();
    unsigned destArgNum = numFuncArgs - numBufferOperands;

    rewriter.setInsertionPoint(returnOp);
    // Find the corresponding destination buffer for each memref operand.
    for (Value operand : operands)
      if (operand.getType().isa<MemRefType>()) {
        assert(destArgNum < numFuncArgs &&
               "The number of operands of return operation is more than the "
               "number of function argument.");

        // For each memref type operand of the source `ReturnOp`, a new `CopyOp`
        // is inserted that copies the buffer content from the operand to the
        // target.
        rewriter.create<CopyOpTy>(loc, operand,
                                  entryBlock.getArgument(destArgNum));
        ++destArgNum;
      }

    // Insert the new target Return operation.
    rewriter.replaceOpWithNewOp<ReturnOpTargetTy>(returnOp, newOperands);
    return success();
  }
};

/// A helper type converter class for using inside Buffer Assignment operation
/// conversion patterns. The default constructor keeps all the types intact
/// except for the ranked-tensor types which is converted to memref types.
class BufferAssignmentTypeConverter : public TypeConverter {
public:
  BufferAssignmentTypeConverter();
};

} // end namespace mlir

#endif // MLIR_TRANSFORMS_BUFFERPLACEMENT_H
