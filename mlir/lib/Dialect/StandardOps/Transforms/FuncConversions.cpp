//===- FuncConversions.cpp - Standard Function conversions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
/// Converts the operand and result types of the Standard's CallOp, used
/// together with the FuncOpSignatureConversion.
struct CallOpSignatureConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult
  matchAndRewrite(CallOp callOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the original function results.
    SmallVector<Type, 1> convertedResults;
    if (failed(typeConverter->convertTypes(callOp.getResultTypes(),
                                           convertedResults)))
      return failure();

    // Substitute with the new result types from the corresponding FuncType
    // conversion.
    rewriter.replaceOpWithNewOp<CallOp>(callOp, callOp.callee(),
                                        convertedResults, operands);
    return success();
  }
};
} // end anonymous namespace

void mlir::populateCallOpTypeConversionPattern(RewritePatternSet &patterns,
                                               TypeConverter &converter) {
  patterns.add<CallOpSignatureConversion>(converter, patterns.getContext());
}

namespace {
/// Only needed to support partial conversion of functions where this pattern
/// ensures that the branch operation arguments matches up with the succesor
/// block arguments.
class BranchOpInterfaceTypeConversion : public ConversionPattern {
public:
  BranchOpInterfaceTypeConversion(TypeConverter &typeConverter,
                                  MLIRContext *ctx)
      : ConversionPattern(/*benefit=*/1, typeConverter, MatchAnyOpTypeTag()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto branchOp = dyn_cast<BranchOpInterface>(op);
    if (!branchOp)
      return failure();

    // For a branch operation, only some operands go to the target blocks, so
    // only rewrite those.
    SmallVector<Value, 4> newOperands(op->operand_begin(), op->operand_end());
    for (int succIdx = 0, succEnd = op->getBlock()->getNumSuccessors();
         succIdx < succEnd; ++succIdx) {
      auto successorOperands = branchOp.getSuccessorOperands(succIdx);
      if (!successorOperands)
        continue;
      for (int idx = successorOperands->getBeginOperandIndex(),
               eidx = idx + successorOperands->size();
           idx < eidx; ++idx) {
        newOperands[idx] = operands[idx];
      }
    }
    rewriter.updateRootInPlace(
        op, [newOperands, op]() { op->setOperands(newOperands); });
    return success();
  }
};
} // end anonymous namespace

namespace {
/// Only needed to support partial conversion of functions where this pattern
/// ensures that the branch operation arguments matches up with the succesor
/// block arguments.
class ReturnOpTypeConversion : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // For a return, all operands go to the results of the parent, so
    // rewrite them all.
    Operation *operation = op.getOperation();
    rewriter.updateRootInPlace(
        op, [operands, operation]() { operation->setOperands(operands); });
    return success();
  }
};
} // end anonymous namespace

void mlir::populateBranchOpInterfaceTypeConversionPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<BranchOpInterfaceTypeConversion>(typeConverter,
                                                patterns.getContext());
}

bool mlir::isLegalForBranchOpInterfaceTypeConversionPattern(
    Operation *op, TypeConverter &converter) {
  // All successor operands of branch like operations must be rewritten.
  if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
    for (int p = 0, e = op->getBlock()->getNumSuccessors(); p < e; ++p) {
      auto successorOperands = branchOp.getSuccessorOperands(p);
      if (successorOperands.hasValue() &&
          !converter.isLegal(successorOperands.getValue().getTypes()))
        return false;
    }
    return true;
  }

  return false;
}

void mlir::populateReturnOpTypeConversionPattern(RewritePatternSet &patterns,
                                                 TypeConverter &typeConverter) {
  patterns.add<ReturnOpTypeConversion>(typeConverter, patterns.getContext());
}

bool mlir::isLegalForReturnOpTypeConversionPattern(Operation *op,
                                                   TypeConverter &converter,
                                                   bool returnOpAlwaysLegal) {
  // If this is a `return` and the user pass wants to convert/transform across
  // function boundaries, then `converter` is invoked to check whether the the
  // `return` op is legal.
  if (dyn_cast<ReturnOp>(op) && !returnOpAlwaysLegal)
    return converter.isLegal(op);

  // ReturnLike operations have to be legalized with their parent. For
  // return this is handled, for other ops they remain as is.
  if (op->hasTrait<OpTrait::ReturnLike>())
    return true;

  return false;
}

bool mlir::isNotBranchOpInterfaceOrReturnLikeOp(Operation *op) {
  // If it is not a terminator, ignore it.
  if (!op->mightHaveTrait<OpTrait::IsTerminator>())
    return true;

  // If it is not the last operation in the block, also ignore it. We do
  // this to handle unknown operations, as well.
  Block *block = op->getBlock();
  if (!block || &block->back() != op)
    return true;

  return false;
}
