//===- OpToFuncCallLowering.h - GPU ops lowering to custom calls *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"

namespace mlir {

/// Rewriting that replace SourceOp with a CallOp to `f32Func` or `f64Func`
/// depending on the element type that Op operates upon. The function
/// declaration is added in case it was not added before.
///
/// If the input values are of f16 type, the value is first casted to f32, the
/// function called and then the result casted back.
///
/// Example with NVVM:
///   %exp_f32 = std.exp %arg_f32 : f32
///
/// will be transformed into
///   llvm.call @__nv_expf(%arg_f32) : (f32) -> f32
template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter &lowering_, StringRef f32Func,
                                StringRef f64Func)
      : ConvertOpToLLVMPattern<SourceOp>(lowering_), f32Func(f32Func),
        f64Func(f64Func) {}

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                  SourceOp>::value,
                  "expected op with same operand and result types");

    SmallVector<Value, 1> castedOperands;
    for (Value operand : operands)
      castedOperands.push_back(maybeCast(operand, rewriter));

    Type resultType = castedOperands.front().getType();
    Type funcType = getFunctionType(resultType, castedOperands);
    StringRef funcName = getFunctionName(
        funcType.cast<LLVM::LLVMFunctionType>().getReturnType());
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), resultType, rewriter.getSymbolRefAttr(funcOp),
        castedOperands);

    if (resultType == operands.front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult(0)});
      return success();
    }

    Value truncated = rewriter.create<LLVM::FPTruncOp>(
        op->getLoc(), operands.front().getType(), callOp.getResult(0));
    rewriter.replaceOp(op, {truncated});
    return success();
  }

private:
  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    Type type = operand.getType();
    if (!type.isa<Float16Type>())
      return operand;

    return rewriter.create<LLVM::FPExtOp>(
        operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  Type getFunctionType(Type resultType, ArrayRef<Value> operands) const {
    SmallVector<Type, 1> operandTypes;
    for (Value operand : operands) {
      operandTypes.push_back(operand.getType());
    }
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  StringRef getFunctionName(Type type) const {
    if (type.isa<Float32Type>())
      return f32Func;
    if (type.isa<Float64Type>())
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcName);
    if (funcOp)
      return cast<LLVMFuncOp>(*funcOp);

    mlir::OpBuilder b(op->getParentOfType<LLVMFuncOp>());
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

  const std::string f32Func;
  const std::string f64Func;
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
