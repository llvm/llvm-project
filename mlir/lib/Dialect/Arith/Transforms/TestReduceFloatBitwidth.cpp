//===- TestReduceFloatBitwdith.cpp - Reduce Float Bitwidth  -*- c++ -----*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that reduces the bitwidth of Arith floating-point IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::arith;

#include "TestReduceFloatBitwidthPatterns.h.inc"

namespace {

/// Options for rewrite patterns.
struct ReduceFloatOptions {
  /// The source float type, who's bit width should be reduced.
  FloatType sourceType;
  /// The target float type.
  FloatType targetType;
};

/// Pattern for arith.addf.
class AddFOpPattern : public OpRewritePattern<AddFOp> {
public:
  AddFOpPattern(MLIRContext *context, const ReduceFloatOptions &options,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(AddFOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType() != options.sourceType)
      return rewriter.notifyMatchFailure(op, "does not match source type");
    Value lhsTrunc =
        rewriter.create<TruncFOp>(op.getLoc(), options.targetType, op.getLhs());
    Value rhsTrunc =
        rewriter.create<TruncFOp>(op.getLoc(), options.targetType, op.getRhs());
    Value newAdd = rewriter.create<AddFOp>(op.getLoc(), lhsTrunc, rhsTrunc);
    rewriter.replaceOpWithNewOp<ExtFOp>(op, op.getType(), newAdd);
    return success();
  }

private:
  const ReduceFloatOptions options;
};
class AddFOpPatternV2 : public OpRewritePattern<AddFOp> {
public:
  AddFOpPatternV2(MLIRContext *context, const ReduceFloatOptions &options,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(AddFOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType() != options.sourceType)
      return rewriter.notifyMatchFailure(op, "does not match source type");
    Value lhsTrunc =
        rewriter.create<TruncFOp>(op.getLoc(), options.targetType, op.getLhs());
    Value rhsTrunc =
        rewriter.create<TruncFOp>(op.getLoc(), options.targetType, op.getRhs());
    Value newAdd = rewriter.create<AddFOp>(op.getLoc(), lhsTrunc, rhsTrunc);
    Value replacementValue =
        rewriter.create<ExtFOp>(op.getLoc(), op.getType(), newAdd);
    rewriter.replaceAllUsesWith(op.getResult(), replacementValue);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const ReduceFloatOptions options;
};

/// Pattern for arith.constant.
class ConstantOpPattern : public OpRewritePattern<ConstantOp> {
public:
  ConstantOpPattern(MLIRContext *context, const ReduceFloatOptions &options,
                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(ConstantOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getType() != options.sourceType)
      return rewriter.notifyMatchFailure(op, "does not match source type");
    double val = cast<FloatAttr>(op.getValue()).getValueAsDouble();
    auto newAttr = FloatAttr::get(options.targetType, val);
    Value newConstant = rewriter.create<ConstantOp>(op.getLoc(), newAttr);
    rewriter.replaceOpWithNewOp<ExtFOp>(op, op.getType(), newConstant);
    return success();
  }

private:
  const ReduceFloatOptions options;
};

/// Pattern for func.func.
class FuncOpPattern : public OpRewritePattern<func::FuncOp> {
public:
  FuncOpPattern(MLIRContext *context, const ReduceFloatOptions &options,
                PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (!llvm::hasSingleElement(op.getBody()))
      return rewriter.notifyMatchFailure(op, "0 or >1 blocks not supported");
    FunctionType type = op.getFunctionType();
    SmallVector<Type> newInputs;
    for (Type t : type.getInputs()) {
      if (t == options.sourceType) {
        newInputs.push_back(options.targetType);
      } else {
        newInputs.push_back(t);
      }
    }
    SmallVector<Type> newResults;
    for (Type t : type.getResults()) {
      if (t == options.sourceType) {
        newResults.push_back(options.targetType);
      } else {
        newResults.push_back(t);
      }
    }
    if (llvm::equal(type.getInputs(), newInputs) &&
        llvm::equal(type.getResults(), newResults))
      return rewriter.notifyMatchFailure(op, "no types to convert");
    auto newFuncOp = rewriter.create<func::FuncOp>(
        op.getLoc(), op.getSymName(),
        FunctionType::get(op.getContext(), newInputs, newResults));
    SmallVector<Location> locs =
        llvm::map_to_vector(op.getBody().getArguments(),
                            [](BlockArgument arg) { return arg.getLoc(); });
    Block *newBlock = rewriter.createBlock(
        &newFuncOp.getBody(), newFuncOp.getBody().begin(), newInputs, locs);
    rewriter.setInsertionPointToStart(newBlock);
    SmallVector<Value> argRepl;
    for (auto [oldType, newType, newArg] : llvm::zip_equal(
             type.getInputs(), newInputs, newBlock->getArguments())) {
      if (oldType == newType) {
        argRepl.push_back(newArg);
      } else {
        argRepl.push_back(
            rewriter.create<ExtFOp>(newArg.getLoc(), oldType, newArg));
      }
    }
    rewriter.inlineBlockBefore(&op.getBody().front(), newBlock, newBlock->end(),
                               argRepl);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const ReduceFloatOptions options;
};

/// Pattern for func.return.
class ReturnOpPattern : public OpRewritePattern<func::ReturnOp> {
public:
  ReturnOpPattern(MLIRContext *context, const ReduceFloatOptions &options,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), options(options) {}

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    bool changedIR = false;
    SmallVector<Value> newOperands;
    for (Value val : op.getOperands()) {
      if (val.getType() != options.sourceType) {
        newOperands.push_back(val);
      } else {
        changedIR = true;
        newOperands.push_back(
            rewriter.create<TruncFOp>(val.getLoc(), options.targetType, val));
      }
    }
    if (!changedIR)
      return rewriter.notifyMatchFailure(op, "no types to convert");
    rewriter.modifyOpInPlace(
        op, [&]() { op.getOperandsMutable().assign(newOperands); });
    return success();
  }

private:
  const ReduceFloatOptions options;
};

/// Pattern that folds arith.truncf(arith.extf(x)) => x.
class ExtTruncFolding : public OpRewritePattern<TruncFOp> {
public:
  ExtTruncFolding(MLIRContext *context, const ReduceFloatOptions &options,
                  PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(TruncFOp op,
                                PatternRewriter &rewriter) const override {
    auto extfOp = op.getIn().getDefiningOp<ExtFOp>();
    if (!extfOp)
      return rewriter.notifyMatchFailure(op,
                                         "'in' is not defined by arith.extf");
    if (extfOp.getIn().getType() != op.getType())
      return rewriter.notifyMatchFailure(op, "types do not match");
    rewriter.replaceOp(op, extfOp.getIn());
    return success();
  }
};

struct TestReduceFloatBitwidthPass
    : public PassWrapper<TestReduceFloatBitwidthPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestReduceFloatBitwidthPass)

  TestReduceFloatBitwidthPass() = default;
  TestReduceFloatBitwidthPass(const TestReduceFloatBitwidthPass &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect>();
  }
  StringRef getArgument() const final {
    return "test-arith-reduce-float-bitwidth";
  }
  StringRef getDescription() const final {
    return "Pass that reduces the bitwidth of floating-point ops";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateTestReduceFloatBitwidthPatterns(
        patterns, optPatterns, FloatType::getF32(ctx), FloatType::getF16(ctx));

    GreedyRewriteConfig config;
    config.fold = optFold;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      getOperation()->emitError() << getArgument() << " failed";
      signalPassFailure();
    }
  }

  Option<bool> optFold{*this, "fold", llvm::cl::init(true),
                       llvm::cl::desc("fold ops")};
  ListOption<std::string> optPatterns{*this, "patterns",
                                      llvm::cl::desc("activated patterns")};
};
} // namespace

static Attribute convertAttrF32ToF16(PatternRewriter &rewriter,
                                     Attribute attr) {
  auto floatAttr = dyn_cast<FloatAttr>(attr);
  if (!attr)
    return Attribute();
  return rewriter.getFloatAttr(rewriter.getF16Type(),
                               floatAttr.getValueAsDouble());
}

void arith::populateTestReduceFloatBitwidthPatterns(
    RewritePatternSet &patterns, ArrayRef<std::string> enabledPatterns,
    FloatType sourceType, FloatType targetType) {
  ReduceFloatOptions options{sourceType, targetType};
  MLIRContext *ctx = patterns.getContext();
  if (llvm::is_contained(enabledPatterns, "arith.addf"))
    patterns.insert<AddFOpPattern>(ctx, options);
  if (llvm::is_contained(enabledPatterns, "arith.addf_v2"))
    patterns.insert<AddFOpPatternV2>(ctx, options);
  if (llvm::is_contained(enabledPatterns, "arith.constant"))
    patterns.insert<ConstantOpPattern>(ctx, options);
  if (llvm::is_contained(enabledPatterns, "func.func"))
    patterns.insert<FuncOpPattern>(ctx, options);
  if (llvm::is_contained(enabledPatterns, "func.return"))
    patterns.insert<ReturnOpPattern>(ctx, options);
  if (llvm::is_contained(enabledPatterns, "arith.truncf"))
    patterns.insert<ExtTruncFolding>(ctx, options);
  if (llvm::is_contained(enabledPatterns, "pdl_patterns")) {
    patterns.getPDLPatterns().registerRewriteFunction("ConvertAttrF32ToF16",
                                                      convertAttrF32ToF16);
    populateGeneratedPDLLPatterns(patterns);
  }
}

namespace mlir {
void registerTestReduceFloatBitwidthPass() {
  PassRegistration<TestReduceFloatBitwidthPass>();
}
} // namespace mlir
