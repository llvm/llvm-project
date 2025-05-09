//===- OpToFuncCallLowering.h - GPU ops lowering to custom calls *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
#define MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

namespace mlir {

namespace {
/// Detection trait tor the `getFastmath` instance method.
template <typename T>
using has_get_fastmath_t = decltype(std::declval<T>().getFastmath());
} // namespace

/// Rewriting that replaces SourceOp with a CallOp to `f32Func` or `f64Func` or
/// `f32ApproxFunc` or `f16Func` or `i32Type` depending on the element type and
/// the fastMathFlag of that Op, if present. The function declaration is added
/// in case it was not added before.
///
/// If the input values are of bf16 type (or f16 type if f16Func is empty), the
/// value is first casted to f32, the function called and then the result casted
/// back.
///
/// Example with NVVM:
///   %exp_f32 = math.exp %arg_f32 : f32
///
/// will be transformed into
///   llvm.call @__nv_expf(%arg_f32) : (f32) -> f32
///
/// If the fastMathFlag attribute of SourceOp is `afn` or `fast`, this Op lowers
/// to the approximate calculation function.
///
/// Also example with NVVM:
///   %exp_f32 = math.exp %arg_f32 fastmath<afn> : f32
///
/// will be transformed into
///   llvm.call @__nv_fast_expf(%arg_f32) : (f32) -> f32
///
/// Final example with NVVM:
///   %pow_f32 = math.fpowi %arg_f32, %arg_i32
///
/// will be transformed into
///   llvm.call @__nv_powif(%arg_f32, %arg_i32) : (f32, i32) -> f32
template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLowering(const LLVMTypeConverter &lowering,
                                StringRef f32Func, StringRef f64Func,
                                StringRef f32ApproxFunc, StringRef f16Func,
                                StringRef i32Func = "",
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(lowering, benefit), f32Func(f32Func),
        f64Func(f64Func), f32ApproxFunc(f32ApproxFunc), f16Func(f16Func),
        i32Func(i32Func) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    bool isResultBool = op->getResultTypes().front().isInteger(1);
    if constexpr (!std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                   SourceOp>::value) {
      assert(op->getNumOperands() > 0 &&
             "expected op to take at least one operand");
      assert((op->getResultTypes().front() == op->getOperand(0).getType() ||
              isResultBool) &&
             "expected op with same operand and result types");
    }

    if (!op->template getParentOfType<FunctionOpInterface>()) {
      return rewriter.notifyMatchFailure(
          op, "expected op to be within a function region");
    }

    SmallVector<Value, 1> castedOperands;
    for (Value operand : adaptor.getOperands())
      castedOperands.push_back(maybeCast(operand, rewriter));

    Type castedOperandType = castedOperands.front().getType();

    // At ABI level, booleans are treated as i32.
    Type resultType =
        isResultBool ? rewriter.getIntegerType(32) : castedOperandType;
    Type funcType = getFunctionType(resultType, castedOperands);
    StringRef funcName = getFunctionName(castedOperandType, op);
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp =
        rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult()});
      return success();
    }

    // Boolean result are mapping to i32 at the ABI level with zero values being
    // interpreted as false and non-zero values being interpreted as true. Since
    // there is no guarantee of a specific value being used to indicate true,
    // compare for inequality with zero (rather than truncate or shift).
    if (isResultBool) {
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), rewriter.getIntegerType(32),
          rewriter.getI32IntegerAttr(0));
      Value truncated = rewriter.create<LLVM::ICmpOp>(
          op->getLoc(), LLVM::ICmpPredicate::ne, callOp.getResult(), zero);
      rewriter.replaceOp(op, {truncated});
      return success();
    }

    assert(callOp.getResult().getType().isF32() &&
           "only f32 types are supposed to be truncated back");
    Value truncated = rewriter.create<LLVM::FPTruncOp>(
        op->getLoc(), adaptor.getOperands().front().getType(),
        callOp.getResult());
    rewriter.replaceOp(op, {truncated});
    return success();
  }

  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    Type type = operand.getType();
    if (!isa<Float16Type, BFloat16Type>(type))
      return operand;

    // If there's an f16 function, no need to cast f16 values.
    if (!f16Func.empty() && isa<Float16Type>(type))
      return operand;

    return rewriter.create<LLVM::FPExtOp>(
        operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    auto funcOp =
        SymbolTable::lookupNearestSymbolFrom<LLVMFuncOp>(op, funcAttr);
    if (funcOp)
      return funcOp;

    auto parentFunc = op->getParentOfType<FunctionOpInterface>();
    assert(parentFunc && "expected there to be a parent function");
    OpBuilder b(parentFunc);
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }

  StringRef getFunctionName(Type type, SourceOp op) const {
    bool useApprox = false;
    if constexpr (llvm::is_detected<has_get_fastmath_t, SourceOp>::value) {
      arith::FastMathFlags flag = op.getFastmath();
      useApprox = ((uint32_t)arith::FastMathFlags::afn & (uint32_t)flag) &&
                  !f32ApproxFunc.empty();
    }

    if (isa<Float16Type>(type))
      return f16Func;
    if (isa<Float32Type>(type)) {
      if (useApprox)
        return f32ApproxFunc;
      return f32Func;
    }
    if (isa<Float64Type>(type))
      return f64Func;

    if (type.isInteger(32))
      return i32Func;
    return "";
  }

  const std::string f32Func;
  const std::string f64Func;
  const std::string f32ApproxFunc;
  const std::string f16Func;
  const std::string i32Func;
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
