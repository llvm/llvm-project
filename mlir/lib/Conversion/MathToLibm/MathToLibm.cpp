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
#define GEN_PASS_DEF_CONVERTMATHTOLIBM
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
  ScalarOpToLibmCall(MLIRContext *context, StringRef floatFunc,
                     StringRef doubleFunc)
      : OpRewritePattern<Op>(context), floatFunc(floatFunc),
        doubleFunc(doubleFunc){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

private:
  std::string floatFunc, doubleFunc;
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns, MLIRContext *ctx,
                           StringRef floatFunc, StringRef doubleFunc) {
  patterns.add<VecOpToScalarOp<OpTy>, PromoteOpToF32<OpTy>>(ctx);
  patterns.add<ScalarOpToLibmCall<OpTy>>(ctx, floatFunc, doubleFunc);
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

void mlir::populateMathToLibmConversionPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  populatePatternsForOp<math::AbsFOp>(patterns, ctx, "fabsf", "fabs");
  populatePatternsForOp<math::AcosOp>(patterns, ctx, "acosf", "acos");
  populatePatternsForOp<math::AcoshOp>(patterns, ctx, "acoshf", "acosh");
  populatePatternsForOp<math::AsinOp>(patterns, ctx, "asinf", "asin");
  populatePatternsForOp<math::AsinhOp>(patterns, ctx, "asinhf", "asinh");
  populatePatternsForOp<math::Atan2Op>(patterns, ctx, "atan2f", "atan2");
  populatePatternsForOp<math::AtanOp>(patterns, ctx, "atanf", "atan");
  populatePatternsForOp<math::AtanhOp>(patterns, ctx, "atanhf", "atanh");
  populatePatternsForOp<math::CbrtOp>(patterns, ctx, "cbrtf", "cbrt");
  populatePatternsForOp<math::CeilOp>(patterns, ctx, "ceilf", "ceil");
  populatePatternsForOp<math::CosOp>(patterns, ctx, "cosf", "cos");
  populatePatternsForOp<math::CoshOp>(patterns, ctx, "coshf", "cosh");
  populatePatternsForOp<math::ErfOp>(patterns, ctx, "erff", "erf");
  populatePatternsForOp<math::ExpOp>(patterns, ctx, "expf", "exp");
  populatePatternsForOp<math::Exp2Op>(patterns, ctx, "exp2f", "exp2");
  populatePatternsForOp<math::ExpM1Op>(patterns, ctx, "expm1f", "expm1");
  populatePatternsForOp<math::FloorOp>(patterns, ctx, "floorf", "floor");
  populatePatternsForOp<math::FmaOp>(patterns, ctx, "fmaf", "fma");
  populatePatternsForOp<math::LogOp>(patterns, ctx, "logf", "log");
  populatePatternsForOp<math::Log2Op>(patterns, ctx, "log2f", "log2");
  populatePatternsForOp<math::Log10Op>(patterns, ctx, "log10f", "log10");
  populatePatternsForOp<math::Log1pOp>(patterns, ctx, "log1pf", "log1p");
  populatePatternsForOp<math::PowFOp>(patterns, ctx, "powf", "pow");
  populatePatternsForOp<math::RoundEvenOp>(patterns, ctx, "roundevenf",
                                           "roundeven");
  populatePatternsForOp<math::RoundOp>(patterns, ctx, "roundf", "round");
  populatePatternsForOp<math::SinOp>(patterns, ctx, "sinf", "sin");
  populatePatternsForOp<math::SinhOp>(patterns, ctx, "sinhf", "sinh");
  populatePatternsForOp<math::SqrtOp>(patterns, ctx, "sqrtf", "sqrt");
  populatePatternsForOp<math::TanOp>(patterns, ctx, "tanf", "tan");
  populatePatternsForOp<math::TanhOp>(patterns, ctx, "tanhf", "tanh");
  populatePatternsForOp<math::TruncOp>(patterns, ctx, "truncf", "trunc");
}

namespace {
struct ConvertMathToLibmPass
    : public impl::ConvertMathToLibmBase<ConvertMathToLibmPass> {
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

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertMathToLibmPass() {
  return std::make_unique<ConvertMathToLibmPass>();
}
