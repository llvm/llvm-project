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

/// This conversion adds an extra argument for each function result which makes
/// the converted function a void function. A type converter must be provided
/// for this conversion to convert a non-shaped type to memref.
/// BufferAssignmentTypeConverter is an helper TypeConverter for this
/// purpose. All the non-shaped type of the input function will be converted to
/// memref.
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

/// This pattern converter transforms a non-void ReturnOpSourceTy into a void
/// return of type ReturnOpTargetTy. It uses a copy operation of type CopyOpTy
/// to copy the results to the output buffer.
template <typename ReturnOpSourceTy, typename ReturnOpTargetTy,
          typename CopyOpTy>
class NonVoidToVoidReturnOpConverter
    : public BufferAssignmentOpConversionPattern<ReturnOpSourceTy> {
public:
  using BufferAssignmentOpConversionPattern<
      ReturnOpSourceTy>::BufferAssignmentOpConversionPattern;

  /// Performs the actual return-op conversion step.
  LogicalResult
  matchAndRewrite(ReturnOpSourceTy returnOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    unsigned numReturnValues = returnOp.getNumOperands();
    Block &entryBlock = returnOp.getParentRegion()->front();
    unsigned numFuncArgs = entryBlock.getNumArguments();
    Location loc = returnOp.getLoc();

    // Find the corresponding output buffer for each operand.
    assert(numReturnValues <= numFuncArgs &&
           "The number of operands of return operation is more than the "
           "number of function argument.");
    unsigned firstReturnParameter = numFuncArgs - numReturnValues;
    for (auto operand : llvm::enumerate(operands)) {
      unsigned returnArgNumber = firstReturnParameter + operand.index();
      BlockArgument dstBuffer = entryBlock.getArgument(returnArgNumber);
      if (dstBuffer == operand.value())
        continue;

      // Insert the copy operation to copy before the return.
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<CopyOpTy>(loc, operand.value(),
                                entryBlock.getArgument(returnArgNumber));
    }
    // Insert the new target return operation.
    rewriter.replaceOpWithNewOp<ReturnOpTargetTy>(returnOp);
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
