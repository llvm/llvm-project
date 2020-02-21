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
/// Example with NVVM:
///   %exp_f32 = std.exp %arg_f32 : f32
///
/// will be transformed into
///   llvm.call @__nv_expf(%arg_f32) : (!llvm.float) -> !llvm.float
template <typename SourceOp>
struct OpToFuncCallLowering : public ConvertToLLVMPattern {
public:
  explicit OpToFuncCallLowering(LLVMTypeConverter &lowering_, StringRef f32Func,
                                StringRef f64Func)
      : ConvertToLLVMPattern(SourceOp::getOperationName(),
                             lowering_.getDialect()->getContext(), lowering_),
        f32Func(f32Func), f64Func(f64Func) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    using LLVM::LLVMFuncOp;
    using LLVM::LLVMType;

    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");

    LLVMType resultType = typeConverter.convertType(op->getResult(0).getType())
                              .template cast<LLVM::LLVMType>();
    LLVMType funcType = getFunctionType(resultType, operands);
    StringRef funcName = getFunctionName(resultType);
    if (funcName.empty())
      return matchFailure();

    LLVMFuncOp funcOp = appendOrGetFuncOp(funcName, funcType, op);
    auto callOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), resultType, rewriter.getSymbolRefAttr(funcOp), operands);
    rewriter.replaceOp(op, {callOp.getResult(0)});
    return matchSuccess();
  }

private:
  LLVM::LLVMType getFunctionType(LLVM::LLVMType resultType,
                                 ArrayRef<Value> operands) const {
    using LLVM::LLVMType;
    SmallVector<LLVMType, 1> operandTypes;
    for (Value operand : operands) {
      operandTypes.push_back(operand.getType().cast<LLVMType>());
    }
    return LLVMType::getFunctionTy(resultType, operandTypes,
                                   /*isVarArg=*/false);
  }

  StringRef getFunctionName(LLVM::LLVMType type) const {
    if (type.isFloatTy())
      return f32Func;
    if (type.isDoubleTy())
      return f64Func;
    return "";
  }

  LLVM::LLVMFuncOp appendOrGetFuncOp(StringRef funcName,
                                     LLVM::LLVMType funcType,
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

namespace gpu {
/// Returns a predicate to be used with addDynamicallyLegalOp. The predicate
/// returns false for calls to the provided intrinsics and true otherwise.
inline std::function<bool(Operation *)>
filterIllegalLLVMIntrinsics(ArrayRef<StringRef> intrinsics, MLIRContext *ctx) {
  SmallVector<StringRef, 4> illegalIds(intrinsics.begin(), intrinsics.end());
  return [illegalIds](Operation *op) -> bool {
    LLVM::CallOp callOp = dyn_cast<LLVM::CallOp>(op);
    if (!callOp || !callOp.callee())
      return true;
    StringRef callee = callOp.callee().getValue();
    return !llvm::any_of(illegalIds, [callee](StringRef intrinsic) {
      return callee.equals(intrinsic);
    });
  };
}
} // namespace gpu

} // namespace mlir

#endif // MLIR_CONVERSION_GPUCOMMON_OPTOFUNCCALLLOWERING_H_
