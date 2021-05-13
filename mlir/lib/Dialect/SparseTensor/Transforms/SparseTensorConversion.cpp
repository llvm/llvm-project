//===- SparseTensorLowering.cpp - Sparse tensor primitives conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Convert sparse tensor primitives to calls into a runtime support library.
// Note that this is a current implementation choice to keep the conversion
// simple. In principle, these primitives could also be converted to actual
// elaborate IR code that implements the primitives on the selected sparse
// tensor storage schemes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

/// Returns internal type encoding for overhead storage.
static unsigned getOverheadTypeEncoding(unsigned width) {
  switch (width) {
  default:
    return 1;
  case 32:
    return 2;
  case 16:
    return 3;
  case 8:
    return 4;
  }
}

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

/// Sparse conversion rule for returns.
class SparseReturnConverter : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, operands);
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

/// Sparse conversion rule for the new operator.
class SparseTensorNewConverter : public OpConversionPattern<NewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NewOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    MLIRContext *context = op->getContext();
    SmallVector<Value, 5> params;
    // Sparse encoding.
    auto enc = getSparseTensorEncoding(resType);
    if (!enc)
      return failure();
    // User pointer.
    params.push_back(operands[0]);
    // Sparsity annotations in tensor constant form. Note that we cast
    // the static shape into a dynamic shape to ensure that the method
    // signature remains uniform accross different tensor dimensions.
    SmallVector<bool, 4> attrs;
    unsigned sz = enc.getDimLevelType().size();
    for (unsigned i = 0; i < sz; i++)
      attrs.push_back(enc.getDimLevelType()[i] ==
                      SparseTensorEncodingAttr::DimLevelType::Compressed);
    Type etp = rewriter.getIntegerType(1);
    RankedTensorType tt1 = RankedTensorType::get({sz}, etp);
    RankedTensorType tt2 =
        RankedTensorType::get({ShapedType::kDynamicSize}, etp);
    auto elts =
        rewriter.create<ConstantOp>(loc, DenseElementsAttr::get(tt1, attrs));
    params.push_back(rewriter.create<tensor::CastOp>(loc, tt2, elts));
    // Seconary and primary types encoding.
    unsigned secPtr = getOverheadTypeEncoding(enc.getPointerBitWidth());
    unsigned secInd = getOverheadTypeEncoding(enc.getIndexBitWidth());
    unsigned primary;
    if (eltType.isF64())
      primary = 1;
    else if (eltType.isF32())
      primary = 2;
    else if (eltType.isInteger(32))
      primary = 3;
    else if (eltType.isInteger(16))
      primary = 4;
    else if (eltType.isInteger(8))
      primary = 5;
    else
      return failure();
    params.push_back(
        rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(secPtr)));
    params.push_back(
        rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(secInd)));
    params.push_back(
        rewriter.create<ConstantOp>(loc, rewriter.getI64IntegerAttr(primary)));
    // Generate the call to create new tensor.
    Type ptrType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    StringRef name = "newSparseTensor";
    rewriter.replaceOpWithNewOp<CallOp>(
        op, ptrType, getFunc(op, name, ptrType, params), params);
    return success();
  }
};

/// Sparse conversion rule for pointer accesses.
class SparseTensorToPointersConverter
    : public OpConversionPattern<ToPointersOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPointersOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex())
      name = "sparsePointers";
    else if (eltType.isInteger(64))
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
class SparseTensorToIndicesConverter : public OpConversionPattern<ToIndicesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToIndicesOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Type eltType = resType.cast<ShapedType>().getElementType();
    StringRef name;
    if (eltType.isIndex())
      name = "sparseIndices";
    else if (eltType.isInteger(64))
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
class SparseTensorToValuesConverter : public OpConversionPattern<ToValuesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToValuesOp op, ArrayRef<Value> operands,
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
void mlir::populateSparseTensorConversionPatterns(TypeConverter &typeConverter,
                                                  RewritePatternSet &patterns) {
  patterns.add<SparseReturnConverter, SparseTensorToDimSizeConverter,
               SparseTensorNewConverter, SparseTensorToPointersConverter,
               SparseTensorToIndicesConverter, SparseTensorToValuesConverter>(
      typeConverter, patterns.getContext());
}
