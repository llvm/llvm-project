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
  ScalarOpToLibmCall<Op>(MLIRContext *context, StringRef floatFunc,
                         StringRef doubleFunc)
      : OpRewritePattern<Op>(context), floatFunc(floatFunc),
        doubleFunc(doubleFunc){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

private:
  std::string floatFunc, doubleFunc;
};
} // namespace

template <typename Op>
LogicalResult
VecOpToScalarOp<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto opType = op.getType();
  auto loc = op.getLoc();
  auto vecType = opType.template dyn_cast<VectorType>();

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
    SmallVector<int64_t> positions = delinearize(strides, linearIndex);
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
  if (!opType.template isa<Float16Type, BFloat16Type>())
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
  if (!type.template isa<Float32Type, Float64Type>())
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
  patterns.add<VecOpToScalarOp<math::Atan2Op>, VecOpToScalarOp<math::CbrtOp>,
               VecOpToScalarOp<math::ExpM1Op>, VecOpToScalarOp<math::TanhOp>,
               VecOpToScalarOp<math::CosOp>, VecOpToScalarOp<math::SinOp>,
               VecOpToScalarOp<math::ErfOp>, VecOpToScalarOp<math::RoundEvenOp>,
               VecOpToScalarOp<math::RoundOp>, VecOpToScalarOp<math::AtanOp>,
               VecOpToScalarOp<math::TanOp>, VecOpToScalarOp<math::TruncOp>>(
      ctx);
  patterns.add<PromoteOpToF32<math::Atan2Op>, PromoteOpToF32<math::CbrtOp>,
               PromoteOpToF32<math::ExpM1Op>, PromoteOpToF32<math::TanhOp>,
               PromoteOpToF32<math::CosOp>, PromoteOpToF32<math::SinOp>,
               PromoteOpToF32<math::ErfOp>, PromoteOpToF32<math::RoundEvenOp>,
               PromoteOpToF32<math::RoundOp>, PromoteOpToF32<math::AtanOp>,
               PromoteOpToF32<math::TanOp>, PromoteOpToF32<math::TruncOp>>(ctx);
  patterns.add<ScalarOpToLibmCall<math::AtanOp>>(ctx, "atanf", "atan");
  patterns.add<ScalarOpToLibmCall<math::Atan2Op>>(ctx, "atan2f", "atan2");
  patterns.add<ScalarOpToLibmCall<math::CbrtOp>>(ctx, "cbrtf", "cbrt");
  patterns.add<ScalarOpToLibmCall<math::ErfOp>>(ctx, "erff", "erf");
  patterns.add<ScalarOpToLibmCall<math::ExpM1Op>>(ctx, "expm1f", "expm1");
  patterns.add<ScalarOpToLibmCall<math::TanOp>>(ctx, "tanf", "tan");
  patterns.add<ScalarOpToLibmCall<math::TanhOp>>(ctx, "tanhf", "tanh");
  patterns.add<ScalarOpToLibmCall<math::RoundEvenOp>>(ctx, "roundevenf",
                                                      "roundeven");
  patterns.add<ScalarOpToLibmCall<math::RoundOp>>(ctx, "roundf", "round");
  patterns.add<ScalarOpToLibmCall<math::CosOp>>(ctx, "cosf", "cos");
  patterns.add<ScalarOpToLibmCall<math::SinOp>>(ctx, "sinf", "sin");
  patterns.add<ScalarOpToLibmCall<math::Log1pOp>>(ctx, "log1pf", "log1p");
  patterns.add<ScalarOpToLibmCall<math::FloorOp>>(ctx, "floorf", "floor");
  patterns.add<ScalarOpToLibmCall<math::CeilOp>>(ctx, "ceilf", "ceil");
  patterns.add<ScalarOpToLibmCall<math::TruncOp>>(ctx, "truncf", "trunc");
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
