//===- AGXToAGXIR.cpp - AGX to AGXIR dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers the `agx` dialect to the AGXIR intrinsic wrappers, like
// AMDGPUToROCDL: shuffles of any type are split into 32-bit parts,
// reductions get their intrinsic operation numbers and 16-bit values are
// extended.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AGXToAGXIR/AGXToAGXIR.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AGX/IR/AGXDialect.h"
#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTAGXTOAGXIRPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::agx;

namespace {

struct ThreadgroupBarrierOpLowering
    : ConvertOpToLLVMPattern<ThreadgroupBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ThreadgroupBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    LLVM::FenceOp::create(rewriter, loc, LLVM::AtomicOrdering::release,
                          "workgroup");
    AGXIR::ThreadgroupBarrierOp::create(rewriter, loc);
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
        op, LLVM::AtomicOrdering::acquire, "workgroup");
    return success();
  }
};

/// Create the AGXIR shuffle for the mode and scope.
static Value createShuffle(ConversionPatternRewriter &rewriter, Location loc,
                           ShuffleMode mode, Scope scope, Value value,
                           Value index) {
  Type type = value.getType();
  bool quad = scope == Scope::quad;
  switch (mode) {
  case ShuffleMode::idx:
    return quad ? AGXIR::QuadShuffleOp::create(rewriter, loc, type, value,
                                               index)
                      .getResult()
                : AGXIR::SIMDShuffleOp::create(rewriter, loc, type, value,
                                               index)
                      .getResult();
  case ShuffleMode::xor_:
    return quad ? AGXIR::QuadShuffleXorOp::create(rewriter, loc, type, value,
                                                  index)
                      .getResult()
                : AGXIR::SIMDShuffleXorOp::create(rewriter, loc, type, value,
                                                  index)
                      .getResult();
  case ShuffleMode::up:
    return quad ? AGXIR::QuadShuffleUpOp::create(rewriter, loc, type, value,
                                                 index)
                      .getResult()
                : AGXIR::SIMDShuffleUpOp::create(rewriter, loc, type, value,
                                                 index)
                      .getResult();
  case ShuffleMode::down:
    return quad ? AGXIR::QuadShuffleDownOp::create(rewriter, loc, type, value,
                                                   index)
                      .getResult()
                : AGXIR::SIMDShuffleDownOp::create(rewriter, loc, type, value,
                                                   index)
                      .getResult();
  }
  llvm_unreachable("unknown shuffle mode");
}

struct ShuffleOpLowering : ConvertOpToLLVMPattern<ShuffleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = adaptor.getValue();
    Type type = value.getType();
    // 16 and 32-bit integers and floats are shuffled directly.
    if (type.isInteger(16) || type.isInteger(32) || type.isF16() ||
        type.isF32()) {
      rewriter.replaceOp(op, createShuffle(rewriter, loc, op.getMode(),
                                           op.getScope(), value,
                                           adaptor.getIndex()));
      return success();
    }
    if (type.isBF16()) {
      Value bits = LLVM::BitcastOp::create(rewriter, loc,
                                           rewriter.getI16Type(), value);
      Value shuffled = createShuffle(rewriter, loc, op.getMode(),
                                     op.getScope(), bits, adaptor.getIndex());
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, type, shuffled);
      return success();
    }
    // Everything else, in 32-bit parts.
    Type i32 = rewriter.getI32Type();
    SmallVector<Value> parts;
    if (failed(LLVM::decomposeValue(rewriter, loc, value, i32, parts)))
      return rewriter.notifyMatchFailure(op, "cannot split value into i32");
    SmallVector<Value> shuffled;
    for (Value part : parts)
      shuffled.push_back(createShuffle(rewriter, loc, op.getMode(),
                                       op.getScope(), part,
                                       adaptor.getIndex()));
    rewriter.replaceOp(op, LLVM::composeValue(rewriter, loc, shuffled, type));
    return success();
  }
};

static AGXIR::ReduceKind getIntrinsicKind(ReduceKind kind, bool isFloat) {
  switch (kind) {
  case ReduceKind::add:
    return isFloat ? AGXIR::ReduceKind::fadd : AGXIR::ReduceKind::iadd;
  case ReduceKind::mul:
    return AGXIR::ReduceKind::fmul;
  case ReduceKind::minsi:
    return AGXIR::ReduceKind::smin;
  case ReduceKind::maxsi:
    return AGXIR::ReduceKind::smax;
  case ReduceKind::minui:
    return AGXIR::ReduceKind::umin;
  case ReduceKind::maxui:
    return AGXIR::ReduceKind::umax;
  case ReduceKind::minnumf:
    return AGXIR::ReduceKind::fmin;
  case ReduceKind::maxnumf:
    return AGXIR::ReduceKind::fmax;
  case ReduceKind::and_:
    return AGXIR::ReduceKind::and_;
  case ReduceKind::or_:
    return AGXIR::ReduceKind::or_;
  case ReduceKind::xor_:
    return AGXIR::ReduceKind::xor_;
  }
  llvm_unreachable("unknown reduction");
}

struct ReduceOpLowering : ConvertOpToLLVMPattern<ReduceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value value = adaptor.getValue();
    Type type = value.getType();
    bool isFloat = isa<FloatType>(type);

    // The hardware reduces 32-bit values: extend 16-bit ones.
    Type wideType = isFloat ? Type(rewriter.getF32Type())
                            : Type(rewriter.getI32Type());
    if (type != wideType) {
      bool isSigned = op.getKind() == ReduceKind::minsi ||
                      op.getKind() == ReduceKind::maxsi;
      if (isFloat)
        value = LLVM::FPExtOp::create(rewriter, loc, wideType, value);
      else if (isSigned)
        value = LLVM::SExtOp::create(rewriter, loc, wideType, value);
      else
        value = LLVM::ZExtOp::create(rewriter, loc, wideType, value);
    }

    AGXIR::ReduceKind kind = getIntrinsicKind(op.getKind(), isFloat);
    Value result;
    bool quad = op.getScope() == Scope::quad;
    if (op.getPrefix())
      result = quad ? AGXIR::QuadPrefixOp::create(rewriter, loc, wideType,
                                                  value, kind)
                          .getResult()
                    : AGXIR::SIMDPrefixOp::create(rewriter, loc, wideType,
                                                  value, kind)
                          .getResult();
    else
      result = quad ? AGXIR::QuadReduceOp::create(rewriter, loc, wideType,
                                                  value, kind)
                          .getResult()
                    : AGXIR::SIMDReduceOp::create(rewriter, loc, wideType,
                                                  value, kind)
                          .getResult();

    if (type != wideType) {
      if (isFloat)
        result = LLVM::FPTruncOp::create(rewriter, loc, type, result);
      else
        result = LLVM::TruncOp::create(rewriter, loc, type, result);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct BallotOpLowering : ConvertOpToLLVMPattern<BallotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BallotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type i32 = rewriter.getI32Type();
    if (op.getScope() == Scope::quad)
      rewriter.replaceOpWithNewOp<AGXIR::QuadBallotOp>(
          op, i32, adaptor.getPredicate());
    else
      rewriter.replaceOpWithNewOp<AGXIR::SIMDBallotOp>(
          op, i32, adaptor.getPredicate());
    return success();
  }
};

struct MemoryBarrierOpLowering : ConvertOpToLLVMPattern<MemoryBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemoryBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // A sequentially consistent fence, which the backend lowers to a memory
    // barrier of its scope.
    StringRef scope = op.getScope() == MemoryScope::threadgroup ? "workgroup" : "";
    rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
        op, LLVM::AtomicOrdering::seq_cst, scope);
    return success();
  }
};

struct MMAOpLowering : ConvertOpToLLVMPattern<MMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AGXIR::SIMDGroupMatrixFMAOp>(
        op, adaptor.getA().getType(), adaptor.getA(), adaptor.getB(),
        adaptor.getC());
    return success();
  }
};

struct SetFPModeOpLowering : ConvertOpToLLVMPattern<SetFPModeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SetFPModeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AGXIR::SetFPModeOp>(op, adaptor.getMode());
    return success();
  }
};

struct GetFPModeOpLowering : ConvertOpToLLVMPattern<GetFPModeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetFPModeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AGXIR::GetFPModeOp>(op, rewriter.getI32Type());
    return success();
  }
};

struct ConvertAGXToAGXIRPass
    : public impl::ConvertAGXToAGXIRPassBase<ConvertAGXToAGXIRPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LLVMTypeConverter converter(ctx);
    RewritePatternSet patterns(ctx);
    populateAGXToAGXIRConversionPatterns(converter, patterns);
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<agx::AGXDialect>();
    target.addLegalDialect<AGXIR::AGXIRDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::populateAGXToAGXIRConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ThreadgroupBarrierOpLowering, MemoryBarrierOpLowering,
               MMAOpLowering, SetFPModeOpLowering, GetFPModeOpLowering,
               ShuffleOpLowering,
               ReduceOpLowering, BallotOpLowering>(converter);
}
