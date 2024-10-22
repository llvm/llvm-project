
//===- MathToEmitC.cpp - Math to EmitC Pass Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MathToEmitC/MathToEmitC.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMATHTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace {

//  Replaces Math operations with `emitc.call_opaque` operations.
struct ConvertMathToEmitCPass
    : public impl::ConvertMathToEmitCBase<ConvertMathToEmitCPass> {
public:
  void runOnOperation() final;
};

} // end anonymous namespace

template <typename OpType>
class LowerToEmitCCallOpaque : public mlir::OpRewritePattern<OpType> {
  std::string calleeStr;

public:
  LowerToEmitCCallOpaque(MLIRContext *context, std::string calleeStr)
      : OpRewritePattern<OpType>(context), calleeStr(calleeStr) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override;
};

// Populates patterns to replace `math` operations with `emitc.call_opaque`,
// using function names consistent with those in <math.h>.
static void populateConvertMathToEmitCPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.insert<LowerToEmitCCallOpaque<math::FloorOp>>(context, "floor");
  patterns.insert<LowerToEmitCCallOpaque<math::RoundEvenOp>>(context, "rint");
  patterns.insert<LowerToEmitCCallOpaque<math::ExpOp>>(context, "exp");
  patterns.insert<LowerToEmitCCallOpaque<math::CosOp>>(context, "cos");
  patterns.insert<LowerToEmitCCallOpaque<math::SinOp>>(context, "sin");
  patterns.insert<LowerToEmitCCallOpaque<math::AcosOp>>(context, "acos");
  patterns.insert<LowerToEmitCCallOpaque<math::AsinOp>>(context, "asin");
  patterns.insert<LowerToEmitCCallOpaque<math::Atan2Op>>(context, "atan2");
  patterns.insert<LowerToEmitCCallOpaque<math::CeilOp>>(context, "ceil");
  patterns.insert<LowerToEmitCCallOpaque<math::AbsFOp>>(context, "fabs");
  patterns.insert<LowerToEmitCCallOpaque<math::FPowIOp>>(context, "powf");
  patterns.insert<LowerToEmitCCallOpaque<math::IPowIOp>>(context, "pow");
}

template <typename OpType>
LogicalResult LowerToEmitCCallOpaque<OpType>::matchAndRewrite(
    OpType op, PatternRewriter &rewriter) const {
  mlir::StringAttr callee = rewriter.getStringAttr(calleeStr);
  auto actualOp = mlir::cast<OpType>(op);
  rewriter.replaceOpWithNewOp<mlir::emitc::CallOpaqueOp>(
      actualOp, actualOp.getType(), callee, actualOp->getOperands());
  return mlir::success();
}

void ConvertMathToEmitCPass::runOnOperation() {
  auto moduleOp = getOperation();
  // Insert #include <math.h> at the beginning of the module
  OpBuilder builder(moduleOp.getBodyRegion());
  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  builder.create<emitc::IncludeOp>(moduleOp.getLoc(),
                                   builder.getStringAttr("math.h"));

  ConversionTarget target(getContext());
  target.addLegalOp<emitc::CallOpaqueOp>();

  target.addIllegalOp<math::FloorOp, math::ExpOp, math::RoundEvenOp,
                      math::CosOp, math::SinOp, math::Atan2Op, math::CeilOp,
                      math::AcosOp, math::AsinOp, math::AbsFOp, math::PowFOp,
                      math::FPowIOp, math::IPowIOp>();

  RewritePatternSet patterns(&getContext());
  populateConvertMathToEmitCPatterns(patterns);

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>>
mlir::createConvertMathToEmitCPass() {
  return std::make_unique<ConvertMathToEmitCPass>();
}
