//===-- MathToNVVM.cpp - conversion from Math to CUDA libdevice calls ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToNVVM/MathToNVVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "../GPUCommon/GPUOpsLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTONVVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "math-to-nvvm"

template <typename OpTy>
static void populateOpPatterns(const LLVMTypeConverter &converter,
                               RewritePatternSet &patterns,
                               PatternBenefit benefit, StringRef f32Func,
                               StringRef f64Func, StringRef f32ApproxFunc = "",
                               StringRef f16Func = "") {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter, benefit);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func,
                                           f32ApproxFunc, f16Func,
                                           /*i32Func=*/"", benefit);
}

template <typename OpTy>
static void populateIntOpPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit, StringRef i32Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter, benefit);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, "", "", "", "", i32Func,
                                           benefit);
}

template <typename OpTy>
static void populateFloatIntOpPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit,
                                       StringRef f32Func, StringRef f64Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter, benefit);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func, "", "",
                                           /*i32Func=*/"", benefit);
}

// Custom pattern for sincos since it returns two values
struct SincosOpLowering : public ConvertOpToLLVMPattern<math::SincosOp> {
  using ConvertOpToLLVMPattern<math::SincosOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::SincosOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getOperand();
    Type inputType = input.getType();
    auto convertedInput = maybeExt(input, rewriter);
    auto computeType = convertedInput.getType();

    StringRef sincosFunc;
    if (isa<Float32Type>(computeType)) {
      const arith::FastMathFlags flag = op.getFastmath();
      const bool useApprox =
          mlir::arith::bitEnumContainsAny(flag, arith::FastMathFlags::afn);
      sincosFunc = useApprox ? "__nv_fast_sincosf" : "__nv_sincosf";
    } else if (isa<Float64Type>(computeType)) {
      sincosFunc = "__nv_sincos";
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported operand type for sincos");
    }

    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    Value sinPtr, cosPtr;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto *scope =
          op->getParentWithTrait<mlir::OpTrait::AutomaticAllocationScope>();
      assert(scope && "Expected op to be inside automatic allocation scope");
      rewriter.setInsertionPointToStart(&scope->getRegion(0).front());
      auto one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(1));
      sinPtr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, computeType, one, 0);
      cosPtr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, computeType, one, 0);
    }

    createSincosCall(rewriter, loc, sincosFunc, convertedInput, sinPtr, cosPtr,
                     op);

    auto sinResult = LLVM::LoadOp::create(rewriter, loc, computeType, sinPtr);
    auto cosResult = LLVM::LoadOp::create(rewriter, loc, computeType, cosPtr);

    rewriter.replaceOp(op, {maybeTrunc(sinResult, inputType, rewriter),
                            maybeTrunc(cosResult, inputType, rewriter)});
    return success();
  }

private:
  Value maybeExt(Value operand, PatternRewriter &rewriter) const {
    if (isa<Float16Type, BFloat16Type>(operand.getType()))
      return LLVM::FPExtOp::create(rewriter, operand.getLoc(),
                                   Float32Type::get(rewriter.getContext()),
                                   operand);
    return operand;
  }

  Value maybeTrunc(Value operand, Type type, PatternRewriter &rewriter) const {
    if (operand.getType() != type)
      return LLVM::FPTruncOp::create(rewriter, operand.getLoc(), type, operand);
    return operand;
  }

  void createSincosCall(ConversionPatternRewriter &rewriter, Location loc,
                        StringRef funcName, Value input, Value sinPtr,
                        Value cosPtr, Operation *op) const {
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto ptrType = sinPtr.getType();

    SmallVector<Type> operandTypes = {input.getType(), ptrType, ptrType};
    auto funcType = LLVM::LLVMFunctionType::get(voidType, operandTypes);

    auto funcAttr = StringAttr::get(op->getContext(), funcName);
    auto funcOp =
        SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(op, funcAttr);

    if (!funcOp) {
      auto parentFunc = op->getParentOfType<FunctionOpInterface>();
      assert(parentFunc && "expected there to be a parent function");
      OpBuilder b(parentFunc);

      auto globalloc = loc->findInstanceOfOrUnknown<FileLineColLoc>();
      funcOp = LLVM::LLVMFuncOp::create(b, globalloc, funcName, funcType);
    }

    SmallVector<Value> callOperands = {input, sinPtr, cosPtr};
    LLVM::CallOp::create(rewriter, loc, funcOp, callOperands);
  }
};

void mlir::populateLibDeviceConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  populateOpPatterns<arith::RemFOp>(converter, patterns, benefit, "__nv_fmodf",
                                    "__nv_fmod");
  populateOpPatterns<arith::MaxNumFOp>(converter, patterns, benefit,
                                       "__nv_fmaxf", "__nv_fmax");
  populateOpPatterns<arith::MinNumFOp>(converter, patterns, benefit,
                                       "__nv_fminf", "__nv_fmin");

  populateIntOpPatterns<math::AbsIOp>(converter, patterns, benefit, "__nv_abs");
  populateOpPatterns<math::AbsFOp>(converter, patterns, benefit, "__nv_fabsf",
                                   "__nv_fabs");
  populateOpPatterns<math::AcosOp>(converter, patterns, benefit, "__nv_acosf",
                                   "__nv_acos");
  populateOpPatterns<math::AcoshOp>(converter, patterns, benefit, "__nv_acoshf",
                                    "__nv_acosh");
  populateOpPatterns<math::AsinOp>(converter, patterns, benefit, "__nv_asinf",
                                   "__nv_asin");
  populateOpPatterns<math::AsinhOp>(converter, patterns, benefit, "__nv_asinhf",
                                    "__nv_asinh");
  populateOpPatterns<math::AtanOp>(converter, patterns, benefit, "__nv_atanf",
                                   "__nv_atan");
  populateOpPatterns<math::Atan2Op>(converter, patterns, benefit, "__nv_atan2f",
                                    "__nv_atan2");
  populateOpPatterns<math::AtanhOp>(converter, patterns, benefit, "__nv_atanhf",
                                    "__nv_atanh");
  populateOpPatterns<math::CbrtOp>(converter, patterns, benefit, "__nv_cbrtf",
                                   "__nv_cbrt");
  populateOpPatterns<math::CeilOp>(converter, patterns, benefit, "__nv_ceilf",
                                   "__nv_ceil");
  populateOpPatterns<math::CopySignOp>(converter, patterns, benefit,
                                       "__nv_copysignf", "__nv_copysign");
  populateOpPatterns<math::CosOp>(converter, patterns, benefit, "__nv_cosf",
                                  "__nv_cos", "__nv_fast_cosf");
  populateOpPatterns<math::CoshOp>(converter, patterns, benefit, "__nv_coshf",
                                   "__nv_cosh");
  populateOpPatterns<math::ErfOp>(converter, patterns, benefit, "__nv_erff",
                                  "__nv_erf");
  populateOpPatterns<math::ErfcOp>(converter, patterns, benefit, "__nv_erfcf",
                                   "__nv_erfc");
  populateOpPatterns<math::ExpOp>(converter, patterns, benefit, "__nv_expf",
                                  "__nv_exp", "__nv_fast_expf");
  populateOpPatterns<math::Exp2Op>(converter, patterns, benefit, "__nv_exp2f",
                                   "__nv_exp2");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, benefit, "__nv_expm1f",
                                    "__nv_expm1");
  populateOpPatterns<math::FloorOp>(converter, patterns, benefit, "__nv_floorf",
                                    "__nv_floor");
  populateOpPatterns<math::FmaOp>(converter, patterns, benefit, "__nv_fmaf",
                                  "__nv_fma");
  // Note: libdevice uses a different name for 32-bit finite checking
  populateOpPatterns<math::IsFiniteOp>(converter, patterns, benefit,
                                       "__nv_finitef", "__nv_isfinited");
  populateOpPatterns<math::IsInfOp>(converter, patterns, benefit, "__nv_isinff",
                                    "__nv_isinfd");
  populateOpPatterns<math::IsNaNOp>(converter, patterns, benefit, "__nv_isnanf",
                                    "__nv_isnand");
  populateOpPatterns<math::LogOp>(converter, patterns, benefit, "__nv_logf",
                                  "__nv_log", "__nv_fast_logf");
  populateOpPatterns<math::Log10Op>(converter, patterns, benefit, "__nv_log10f",
                                    "__nv_log10", "__nv_fast_log10f");
  populateOpPatterns<math::Log1pOp>(converter, patterns, benefit, "__nv_log1pf",
                                    "__nv_log1p");
  populateOpPatterns<math::Log2Op>(converter, patterns, benefit, "__nv_log2f",
                                   "__nv_log2", "__nv_fast_log2f");
  populateOpPatterns<math::PowFOp>(converter, patterns, benefit, "__nv_powf",
                                   "__nv_pow", "__nv_fast_powf");
  populateFloatIntOpPatterns<math::FPowIOp>(converter, patterns, benefit,
                                            "__nv_powif", "__nv_powi");
  populateOpPatterns<math::RoundOp>(converter, patterns, benefit, "__nv_roundf",
                                    "__nv_round");
  populateOpPatterns<math::RoundEvenOp>(converter, patterns, benefit,
                                        "__nv_rintf", "__nv_rint");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, benefit, "__nv_rsqrtf",
                                    "__nv_rsqrt");
  populateOpPatterns<math::SinOp>(converter, patterns, benefit, "__nv_sinf",
                                  "__nv_sin", "__nv_fast_sinf");
  populateOpPatterns<math::SinhOp>(converter, patterns, benefit, "__nv_sinhf",
                                   "__nv_sinh");
  populateOpPatterns<math::SqrtOp>(converter, patterns, benefit, "__nv_sqrtf",
                                   "__nv_sqrt");
  populateOpPatterns<math::TanOp>(converter, patterns, benefit, "__nv_tanf",
                                  "__nv_tan", "__nv_fast_tanf");
  populateOpPatterns<math::TanhOp>(converter, patterns, benefit, "__nv_tanhf",
                                   "__nv_tanh");

  // Custom pattern for sincos since it returns two values
  patterns.add<SincosOpLowering>(converter, benefit);
}

namespace {
struct ConvertMathToNVVMPass final
    : impl::ConvertMathToNVVMBase<ConvertMathToNVVMPass> {
  using impl::ConvertMathToNVVMBase<
      ConvertMathToNVVMPass>::ConvertMathToNVVMBase;

  void runOnOperation() override;
};
} // namespace

void ConvertMathToNVVMPass::runOnOperation() {
  auto m = getOperation();
  MLIRContext *ctx = m.getContext();

  RewritePatternSet patterns(&getContext());
  LowerToLLVMOptions options(ctx, DataLayout(m));
  LLVMTypeConverter converter(ctx, options);

  populateLibDeviceConversionPatterns(converter, patterns, /*benefit=*/1);

  ConversionTarget target(getContext());
  target
      .addLegalDialect<BuiltinDialect, func::FuncDialect, vector::VectorDialect,
                       LLVM::LLVMDialect, NVVM::NVVMDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::Exp2Op, LLVM::FAbsOp,
                      LLVM::FCeilOp, LLVM::FFloorOp, LLVM::FRemOp, LLVM::LogOp,
                      LLVM::Log10Op, LLVM::Log2Op, LLVM::PowOp, LLVM::SinOp,
                      LLVM::SqrtOp>();
  if (failed(applyPartialConversion(m, target, std::move(patterns))))
    signalPassFailure();
}
