//===- PartTensorConversion.cpp - Part tensor primitives conversion ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that converts part tensor primitives into calls into a runtime
// support library. Part tensor types are converted into opaque pointers
// to the underlying part storage schemes. The use of opaque pointers
// together with runtime support library keeps the conversion relatively
// simple, but at the expense of IR opacity, which obscures opportunities
// for subsequent optimization of the IR. An alternative is provided by
// the PartTensorCodegen pass.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PartTensor/IR/PartTensor.h"
#include "mlir/Dialect/PartTensor/IR/PartTensorType.h"
#include "mlir/Dialect/PartTensor/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"

#include "CodegenUtils.h"

using namespace mlir;
using namespace mlir::part_tensor;

namespace {

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Maps each part tensor type to an opaque pointer.
static std::optional<Type> convertPartTensorTypes(Type type) {
  if (mlir::part_tensor::getPartTensorEncoding(type) != nullptr)
    return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
  return std::nullopt;
}

static FunctionType getElementalFuncTypeForOp(Operation *op) {
  SmallVector<Type, 1> resultTys(op->getNumResults());
  SmallVector<Type, 2> inputTys(op->getNumOperands());
  std::transform(op->result_type_begin(), op->result_type_end(),
                 resultTys.begin(),
                 [](Type ty) { return getElementTypeOrSelf(ty); });
  std::transform(op->operand_type_begin(), op->operand_type_end(),
                 inputTys.begin(),
                 [](Type ty) { return getElementTypeOrSelf(ty); });
  return FunctionType::get(op->getContext(), inputTys, resultTys);
}

//===----------------------------------------------------------------------===//
// Conversion rules.
//===----------------------------------------------------------------------===//

/// Part conversion rule for position accesses.
class PartTensorGetPartitionsConverter
    : public OpConversionPattern<GetPartitionsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetPartitionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    const Type crdTp = cast<ShapedType>(resType).getElementType();
    Location loc = op->getLoc();
    MemRefType callRetType = mlir::sparse_tensor::get1DMemRefType(crdTp, false);
    SmallVector<Value> operands{adaptor.getOperands()[0]};
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(), "getPartitions", callRetType, operands,
        mlir::sparse_tensor::EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, callRetType, fn, operands)
            .getResult(0);
    if (resType != callRetType)
      callRet = rewriter.create<memref::CastOp>(loc, resType, callRet);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

class PartTensorGetSliceConverter
    : public OpConversionPattern<GetSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Location loc = op->getLoc();
    SmallVector<Value> operands{adaptor.getOperands()[0]};
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(), "getSlice", resType, adaptor.getOperands(),
        mlir::sparse_tensor::EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, resType, fn, adaptor.getOperands())
            .getResult(0);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

class PartTensorGetNumPartitionsConverter
    : public OpConversionPattern<GetNumPartitionsOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(GetNumPartitionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resType = op.getType();
    Location loc = op->getLoc();
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(), "getNumPartitions", resType, adaptor.getOperands(),
        mlir::sparse_tensor::EmitCInterface::On);
    Value callRet =
        rewriter.create<func::CallOp>(loc, resType, fn, adaptor.getOperands())
            .getResult(0);
    rewriter.replaceOp(op, callRet);
    return success();
  }
};

class PartTensorSetSliceConverter
    : public OpConversionPattern<SetSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SetSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
	auto Context = op.getContext();
    auto resType = std::nullopt;
    Location loc = op->getLoc();
    auto fn = mlir::sparse_tensor::getFunc(
        op->getParentOfType<ModuleOp>(), "SetSlice", resType, adaptor.getOperands(),
        mlir::sparse_tensor::EmitCInterface::On);
        rewriter.create<func::CallOp>(loc, resType, fn, adaptor.getOperands());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Part tensor type conversion into opaque pointer.
//===----------------------------------------------------------------------===//

mlir::PartTensorTypeToPtrConverter::PartTensorTypeToPtrConverter() {
  addConversion([](Type type) { return type; });
  addConversion(convertPartTensorTypes);
}

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void mlir::populatePartTensorConversionPatterns(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns) {
  patterns.add<PartTensorGetPartitionsConverter>(typeConverter,
                                                 patterns.getContext());

  patterns.add<PartTensorGetSliceConverter>(typeConverter,
                                                 patterns.getContext());

  patterns.add<PartTensorGetNumPartitionsConverter>(typeConverter,
                                                 patterns.getContext());

  patterns.add<PartTensorSetSliceConverter>(typeConverter,
                                                 patterns.getContext());
}
