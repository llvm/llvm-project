//===-- MathToLibm.cpp - conversion from Math to libm calls ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToLibm/MathToLibm.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOLIBMPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
// Pattern to convert vector operations to scalar operations. This is needed as
// libm calls require scalars.
template <typename Op>
struct VecOpToScalarOp : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};
// Pattern to promote an op of a smaller floating point type to F32.
template <typename Op>
struct PromoteOpToF32 : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};
// Pattern to convert scalar math operations to calls to libm functions.
// Additionally the libm function signatures are declared.
template <typename Op>
struct ScalarOpToLibmCall : public OpRewritePattern<Op> {
public:
  using OpRewritePattern<Op>::OpRewritePattern;
  ScalarOpToLibmCall(MLIRContext *context, PatternBenefit benefit,
                     StringRef floatFunc, StringRef doubleFunc)
      : OpRewritePattern<Op>(context, benefit), floatFunc(floatFunc),
        doubleFunc(doubleFunc) {};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

private:
  std::string floatFunc, doubleFunc;
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns, PatternBenefit benefit,
                           MLIRContext *ctx, StringRef floatFunc,
                           StringRef doubleFunc) {
  patterns.add<VecOpToScalarOp<OpTy>, PromoteOpToF32<OpTy>>(ctx, benefit);
  patterns.add<ScalarOpToLibmCall<OpTy>>(ctx, benefit, floatFunc, doubleFunc);
}

} // namespace

template <typename Op>
LogicalResult
VecOpToScalarOp<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto opType = op.getType();
  auto loc = op.getLoc();
  auto vecType = dyn_cast<VectorType>(opType);

  if (!vecType)
    return failure();
  if (!vecType.hasRank())
    return failure();
  auto shape = vecType.getShape();
  int64_t numElements = vecType.getNumElements();

  Value result = rewriter.create<arith::ConstantOp>(
      loc, DenseElementsAttr::get(
               vecType, FloatAttr::get(vecType.getElementType(), 0.0)));
  SmallVector<int64_t> strides = computeStrides(shape);
  for (auto linearIndex = 0; linearIndex < numElements; ++linearIndex) {
    SmallVector<int64_t> positions = delinearize(linearIndex, strides);
    SmallVector<Value> operands;
    for (auto input : op->getOperands())
      operands.push_back(
          rewriter.create<vector::ExtractOp>(loc, input, positions));
    Value scalarOp =
        rewriter.create<Op>(loc, vecType.getElementType(), operands);
    result =
        rewriter.create<vector::InsertOp>(loc, scalarOp, result, positions);
  }
  rewriter.replaceOp(op, {result});
  return success();
}

template <typename Op>
LogicalResult
PromoteOpToF32<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto opType = op.getType();
  if (!isa<Float16Type, BFloat16Type>(opType))
    return failure();

  auto loc = op.getLoc();
  auto f32 = rewriter.getF32Type();
  auto extendedOperands = llvm::to_vector(
      llvm::map_range(op->getOperands(), [&](Value operand) -> Value {
        return rewriter.create<arith::ExtFOp>(loc, f32, operand);
      }));
  auto newOp = rewriter.create<Op>(loc, f32, extendedOperands);
  rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, opType, newOp);
  return success();
}

template <typename Op>
LogicalResult
ScalarOpToLibmCall<Op>::matchAndRewrite(Op op,
                                        PatternRewriter &rewriter) const {
  auto module = SymbolTable::getNearestSymbolTable(op);
  auto type = op.getType();
  if (!isa<Float32Type, Float64Type>(type))
    return failure();

  auto name = type.getIntOrFloatBitWidth() == 64 ? doubleFunc : floatFunc;
  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
      SymbolTable::lookupSymbolIn(module, name));
  // Forward declare function if it hasn't already been
  if (!opFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());
    auto opFunctionTy = FunctionType::get(
        rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
    opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name,
                                           opFunctionTy);
    opFunc.setPrivate();

    // By definition Math dialect operations imply LLVM's "readnone"
    // function attribute, so we can set it here to provide more
    // optimization opportunities (e.g. LICM) for backends targeting LLVM IR.
    // This will have to be changed, when strict FP behavior is supported
    // by Math dialect.
    opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                    UnitAttr::get(rewriter.getContext()));
  }
  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

  rewriter.replaceOpWithNewOp<func::CallOp>(op, name, op.getType(),
                                            op->getOperands());

  return success();
}

void mlir::populateMathToLibmConversionPatterns(RewritePatternSet &patterns,
                                                PatternBenefit benefit) {
  MLIRContext *ctx = patterns.getContext();

  populatePatternsForOp<math::AbsFOp>(patterns, benefit, ctx, "fabsf", "fabs");
  populatePatternsForOp<math::AcosOp>(patterns, benefit, ctx, "acosf", "acos");
  populatePatternsForOp<math::AcoshOp>(patterns, benefit, ctx, "acoshf",
                                       "acosh");
  populatePatternsForOp<math::AsinOp>(patterns, benefit, ctx, "asinf", "asin");
  populatePatternsForOp<math::AsinhOp>(patterns, benefit, ctx, "asinhf",
                                       "asinh");
  populatePatternsForOp<math::Atan2Op>(patterns, benefit, ctx, "atan2f",
                                       "atan2");
  populatePatternsForOp<math::AtanOp>(patterns, benefit, ctx, "atanf", "atan");
  populatePatternsForOp<math::AtanhOp>(patterns, benefit, ctx, "atanhf",
                                       "atanh");
  populatePatternsForOp<math::CbrtOp>(patterns, benefit, ctx, "cbrtf", "cbrt");
  populatePatternsForOp<math::CeilOp>(patterns, benefit, ctx, "ceilf", "ceil");
  populatePatternsForOp<math::CosOp>(patterns, benefit, ctx, "cosf", "cos");
  populatePatternsForOp<math::CoshOp>(patterns, benefit, ctx, "coshf", "cosh");
  populatePatternsForOp<math::ErfOp>(patterns, benefit, ctx, "erff", "erf");
  populatePatternsForOp<math::ErfcOp>(patterns, benefit, ctx, "erfcf", "erfc");
  populatePatternsForOp<math::ExpOp>(patterns, benefit, ctx, "expf", "exp");
  populatePatternsForOp<math::Exp2Op>(patterns, benefit, ctx, "exp2f", "exp2");
  populatePatternsForOp<math::ExpM1Op>(patterns, benefit, ctx, "expm1f",
                                       "expm1");
  populatePatternsForOp<math::FloorOp>(patterns, benefit, ctx, "floorf",
                                       "floor");
  populatePatternsForOp<math::FmaOp>(patterns, benefit, ctx, "fmaf", "fma");
  populatePatternsForOp<math::LogOp>(patterns, benefit, ctx, "logf", "log");
  populatePatternsForOp<math::Log2Op>(patterns, benefit, ctx, "log2f", "log2");
  populatePatternsForOp<math::Log10Op>(patterns, benefit, ctx, "log10f",
                                       "log10");
  populatePatternsForOp<math::Log1pOp>(patterns, benefit, ctx, "log1pf",
                                       "log1p");
  populatePatternsForOp<math::PowFOp>(patterns, benefit, ctx, "powf", "pow");
  populatePatternsForOp<math::RoundEvenOp>(patterns, benefit, ctx, "roundevenf",
                                           "roundeven");
  populatePatternsForOp<math::RoundOp>(patterns, benefit, ctx, "roundf",
                                       "round");
  populatePatternsForOp<math::SinOp>(patterns, benefit, ctx, "sinf", "sin");
  populatePatternsForOp<math::SinhOp>(patterns, benefit, ctx, "sinhf", "sinh");
  populatePatternsForOp<math::SqrtOp>(patterns, benefit, ctx, "sqrtf", "sqrt");
  populatePatternsForOp<math::RsqrtOp>(patterns, benefit, ctx, "rsqrtf",
                                       "rsqrt");
  populatePatternsForOp<math::TanOp>(patterns, benefit, ctx, "tanf", "tan");
  populatePatternsForOp<math::TanhOp>(patterns, benefit, ctx, "tanhf", "tanh");
  populatePatternsForOp<math::TruncOp>(patterns, benefit, ctx, "truncf",
                                       "trunc");
}

namespace {
struct ConvertMathToLibmPass
    : public impl::ConvertMathToLibmPassBase<ConvertMathToLibmPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertMathToLibmPass::runOnOperation() {
  auto module = getOperation();

  RewritePatternSet patterns(&getContext());
  populateMathToLibmConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, BuiltinDialect, func::FuncDialect,
                         vector::VectorDialect>();
  target.addIllegalDialect<math::MathDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
