//===- SparseLowering.cpp - Lowers sparse primitives to library calls.  ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

using namespace mlir;

namespace {

/// Returns function reference (first hit also inserts into module).
static FlatSymbolRefAttr getFunc(Operation *op, StringRef name, Type result,
                                 ValueRange operands) {
  MLIRContext *context = op->getContext();
  auto module = op->getParentOfType<ModuleOp>();
  auto func = module.lookupSymbol<FuncOp>(name);
  if (!func) {
    OpBuilder moduleBuilder(module.getBodyRegion());
    moduleBuilder
        .create<FuncOp>(op->getLoc(), name,
                        FunctionType::get(context, operands.getTypes(), result))
        .setPrivate();
  }
  return SymbolRefAttr::get(context, name);
}

/// Sparse conversion rule to remove opaque pointer cast.
class TensorFromPointerConverter
    : public OpConversionPattern<linalg::SparseTensorFromPointerOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::SparseTensorFromPointerOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

/// Sparse conversion rule for dimension accesses.
class TensorToDimSizeConverter : public OpConversionPattern<memref::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::DimOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!operands[0].getType().isa<LLVM::LLVMPointerType>())
      return failure();
    Type resType = op.getType();
    StringRef name = "sparseDimSize";
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for pointer accesses.
class TensorToPointersConverter
    : public OpConversionPattern<linalg::SparseTensorToPointersMemRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::SparseTensorToPointersMemRefOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex() || eltType.isInteger(64))
      name = "sparsePointers64";
    else if (eltType.isInteger(32))
      name = "sparsePointers32";
    else
      return failure();
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for index accesses.
class TensorToIndicesConverter
    : public OpConversionPattern<linalg::SparseTensorToIndicesMemRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::SparseTensorToIndicesMemRefOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex() || eltType.isInteger(64))
      name = "sparseIndices64";
    else if (eltType.isInteger(32))
      name = "sparseIndices32";
    else
      return failure();
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for value accesses.
class TensorToValuesConverter
    : public OpConversionPattern<linalg::SparseTensorToValuesMemRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::SparseTensorToValuesMemRefOp op,
                  ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isF64())
      name = "sparseValuesF64";
    else if (eltType.isF32())
      name = "sparseValuesF32";
    else
      return failure();
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

} // namespace

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void linalg::populateSparsificationConversionPatterns(
    OwningRewritePatternList &patterns) {
  patterns.insert<TensorFromPointerConverter, TensorToDimSizeConverter,
                  TensorToPointersConverter, TensorToIndicesConverter,
                  TensorToValuesConverter>(patterns.getContext());
}
