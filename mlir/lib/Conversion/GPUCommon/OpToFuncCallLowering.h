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

template <typename SourceOp, typename DerivedTy>
struct OpToFuncCallLoweringBase : public ConvertOpToLLVMPattern<SourceOp> {
public:
  explicit OpToFuncCallLoweringBase(const LLVMTypeConverter &lowering)
      : ConvertOpToLLVMPattern<SourceOp>(lowering) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    if constexpr (!std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                   SourceOp>::value) {
      assert(op->getNumOperands() > 0 &&
             "expected op to take at least one operand");
      assert(op->getResultTypes().front() == op->getOperand(0).getType() &&
             "expected op with same operand and result types");
    }

    if (!op->template getParentOfType<FunctionOpInterface>()) {
      return rewriter.notifyMatchFailure(
          op, "expected op to be within a function region");
    }

    SmallVector<Value, 1> castedOperands;
    for (Value operand : adaptor.getOperands())
      castedOperands.push_back(
          static_cast<const DerivedTy *>(this)->maybeCast(operand, rewriter));

    Type resultType = castedOperands.front().getType();
    Type funcType = getFunctionType(resultType, castedOperands);
    StringRef funcName =
        static_cast<const DerivedTy *>(this)
            ->getFunctionName(
                cast<LLVM::LLVMFunctionType>(funcType).getReturnType(), op);
    if (funcName.empty())
      return failure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp =
        rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, castedOperands);

    if (resultType == adaptor.getOperands().front().getType()) {
      rewriter.replaceOp(op, {callOp.getResult()});
      return success();
    }

    Value truncated = rewriter.create<LLVM::FPTruncOp>(
        op->getLoc(), adaptor.getOperands().front().getType(),
        callOp.getResult());
    rewriter.replaceOp(op, {truncated});
    return success();
  }

private:
  Type getFunctionType(Type resultType, ValueRange operands) const {
    SmallVector<Type> operandTypes(operands.getTypes());
    return LLVM::LLVMFunctionType::get(resultType, operandTypes);
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName, Type funcType,
                                     Operation *op) const {
    using LLVM::LLVMFuncOp;

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    auto funcOp = SymbolTable::lookupNearestSymbolFrom<LLVMFuncOp>(op, funcAttr);
    if (funcOp)
      return funcOp;

    auto parentFunc = op->getParentOfType<FunctionOpInterface>();
    assert(parentFunc && "expected there to be a parent function");
    OpBuilder b(parentFunc);
    return b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  }
};

/// Rewriting that replaces SourceOp with a CallOp to `f32Func` or `f64Func` or
/// `f32ApproxFunc` or `f16Func` depending on the element type and the
/// fastMathFlag of that Op. The function declaration is added in case it was
/// not added before.
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
template <typename SourceOp>
struct OpToFuncCallLowering
    : public OpToFuncCallLoweringBase<SourceOp,
                                      OpToFuncCallLowering<SourceOp>> {
public:
  explicit OpToFuncCallLowering(const LLVMTypeConverter &lowering,
                                StringRef f32Func, StringRef f64Func,
                                StringRef f32ApproxFunc, StringRef f16Func)
      : OpToFuncCallLoweringBase<SourceOp, OpToFuncCallLowering<SourceOp>>(
            lowering),
        f32Func(f32Func), f64Func(f64Func), f32ApproxFunc(f32ApproxFunc),
        f16Func(f16Func) {}

  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    Type type = operand.getType();
    if (!isa<Float16Type, BFloat16Type>(type))
      return operand;

    // if there's a f16 function, no need to cast f16 values
    if (!f16Func.empty() && isa<Float16Type>(type))
      return operand;

    return rewriter.create<LLVM::FPExtOp>(
        operand.getLoc(), Float32Type::get(rewriter.getContext()), operand);
  }

  StringRef getFunctionName(Type type, SourceOp op) const {
    arith::FastMathFlags flag = op.getFastmath();
    if (isa<Float16Type>(type))
      return f16Func;
    if (isa<Float32Type>(type)) {
      if (((uint32_t)arith::FastMathFlags::afn & (uint32_t)flag) &&
          !f32ApproxFunc.empty())
        return f32ApproxFunc;
      else
        return f32Func;
    }
    if (isa<Float64Type>(type))
      return f64Func;
    return "";
  }

  const std::string f32Func;
  const std::string f64Func;
  const std::string f32ApproxFunc;
  const std::string f16Func;
};

/// Rewriting that replace SourceOp with a CallOp to `i32Func`
/// The function declaration is added in case it was not added before.
/// This assumes that all types integral.
///
/// Example with NVVM:
///   %abs_i32 = math.iabs %arg_i32 : i32
///
/// will be transformed into
///   llvm.call @__nv_abs(%arg_i32) : (i32) -> i32
///
template <typename SourceOp>
struct IntOpToFuncCallLowering
    : public OpToFuncCallLoweringBase<SourceOp,
                                      IntOpToFuncCallLowering<SourceOp>> {
public:
  explicit IntOpToFuncCallLowering(const LLVMTypeConverter &lowering,
                                   StringRef i32Func)
      : OpToFuncCallLoweringBase<SourceOp, IntOpToFuncCallLowering<SourceOp>>(
            lowering),
        i32Func(i32Func) {}

  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    return operand;
  }

  StringRef getFunctionName(Type type, SourceOp op) const {
    IntegerType itype = dyn_cast<IntegerType>(type);
    if (!itype || itype.getWidth() != 32)
      return "";
    return i32Func;
  }

  const std::string i32Func;
};

/// Rewriting that replaces SourceOp with a CallOp to `f32Func` or `f64Func`,
/// depending on the type of the result. This assumes that the first argument is
/// a floating type and the second argument is an integer type.
///
/// Example with NVVM:
///   %result32 = math.fpowi %arg_f32, %arg_i32 : f32, i32
///
/// will be transformed into
///   llvm.call @__nv_powf(%arg_f32, %arg_i32) : (f32, i32) -> f32
///
template <typename SourceOp>
struct FloatIntOpToFuncCallLowering
    : public OpToFuncCallLoweringBase<SourceOp,
                                      FloatIntOpToFuncCallLowering<SourceOp>> {
public:
  explicit FloatIntOpToFuncCallLowering(const LLVMTypeConverter &lowering,
                                        StringRef f32Func, StringRef f64Func)
      : OpToFuncCallLoweringBase<SourceOp,
                                 FloatIntOpToFuncCallLowering<SourceOp>>(
            lowering),
        f32Func(f32Func), f64Func(f64Func) {}

  Value maybeCast(Value operand, PatternRewriter &rewriter) const {
    return operand;
  }

  StringRef getFunctionName(Type type, SourceOp op) const {
    if (isa<Float32Type>(type)) {
      return f32Func;
    }
    if (isa<Float64Type>(type))
      return f64Func;
    return "";
  }

  const std::string f32Func;
  const std::string f64Func;
};

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
