//===- SparseTensorLowering.cpp - Sparse tensor primitives lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower sparse tensor primitives to calls into a runtime support library.
// Note that this is a current implementation choice to keep the lowering
// simple. In principle, these primitives could also be lowered to actual
// elaborate IR code that implements the primitives on the selected sparse
// tensor storage schemes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

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
class SparseTensorFromPointerConverter
    : public OpConversionPattern<sparse_tensor::FromPointerOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparse_tensor::FromPointerOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, operands[0]);
    return success();
  }
};

/// Sparse conversion rule for dimension accesses.
class SparseTensorToDimSizeConverter
    : public OpConversionPattern<memref::DimOp> {
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
class SparseTensorToPointersConverter
    : public OpConversionPattern<sparse_tensor::ToPointersOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparse_tensor::ToPointersOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex() || eltType.isInteger(64))
      name = "sparsePointers64";
    else if (eltType.isInteger(32))
      name = "sparsePointers32";
    else if (eltType.isInteger(16))
      name = "sparsePointers16";
    else if (eltType.isInteger(8))
      name = "sparsePointers8";
    else
      return failure();
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for index accesses.
class SparseTensorToIndicesConverter
    : public OpConversionPattern<sparse_tensor::ToIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparse_tensor::ToIndicesOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex() || eltType.isInteger(64))
      name = "sparseIndices64";
    else if (eltType.isInteger(32))
      name = "sparseIndices32";
    else if (eltType.isInteger(16))
      name = "sparseIndices16";
    else if (eltType.isInteger(8))
      name = "sparseIndices8";
    else
      return failure();
    rewriter.replaceOpWithNewOp<CallOp>(
        op, resType, getFunc(op, name, resType, operands), operands);
    return success();
  }
};

/// Sparse conversion rule for value accesses.
class SparseTensorToValuesConverter
    : public OpConversionPattern<sparse_tensor::ToValuesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(sparse_tensor::ToValuesOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isF64())
      name = "sparseValuesF64";
    else if (eltType.isF32())
      name = "sparseValuesF32";
    else if (eltType.isInteger(32))
      name = "sparseValuesI32";
    else if (eltType.isInteger(16))
      name = "sparseValuesI16";
    else if (eltType.isInteger(8))
      name = "sparseValuesI8";
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
void mlir::populateSparseTensorConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<SparseTensorFromPointerConverter, SparseTensorToDimSizeConverter,
               SparseTensorToPointersConverter, SparseTensorToIndicesConverter,
               SparseTensorToValuesConverter>(patterns.getContext());
}
