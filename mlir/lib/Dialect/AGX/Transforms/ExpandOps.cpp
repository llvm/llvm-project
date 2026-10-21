//===- ExpandOps.cpp - Portable forms of agx operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites agx operations without a native lowering on the target chip into
// standard dialects and the basic agx operations. Every pattern checks a
// capability of AGXChipInfo, never a generation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AGX/Transforms/Passes.h"

#include "mlir/Dialect/AGX/IR/AGXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::agx {
#define GEN_PASS_DEF_AGXEXPANDOPSPASS
#include "mlir/Dialect/AGX/Transforms/Passes.h.inc"
} // namespace mlir::agx

using namespace mlir;
using namespace mlir::agx;

StringRef mlir::agx::getTargetChip(Operation *op) {
  if (auto module = op->getParentOfType<gpu::GPUModuleOp>())
    if (ArrayAttr targets = module.getTargetsAttr())
      for (Attribute t : targets)
        if (auto target = dyn_cast<AGXIR::AGXIRTargetAttr>(t))
          return target.getChip();
  return "agx-g14g";
}

namespace {

/// The row and column of the first of the two elements thread `lane` holds in
/// a fragment: r = 4 * (t / 16) + (t / 2) % 4, c = 4 * ((t / 8) % 2) + 2 * (t % 2).
std::pair<Value, Value> fragmentPosition(OpBuilder &b, Location loc) {
  Value lane = gpu::LaneIdOp::create(b, loc, /*upper_bound=*/nullptr);
  auto c = [&](int64_t v) { return arith::ConstantIndexOp::create(b, loc, v); };
  auto div = [&](Value x, int64_t d) { return arith::DivUIOp::create(b, loc, x, c(d)); };
  auto rem = [&](Value x, int64_t d) { return arith::RemUIOp::create(b, loc, x, c(d)); };
  auto mul = [&](Value x, int64_t m) { return arith::MulIOp::create(b, loc, x, c(m)); };
  auto add = [&](Value x, Value y) { return arith::AddIOp::create(b, loc, x, y); };
  Value r = add(mul(div(lane, 16), 4), rem(div(lane, 2), 4));
  Value col = add(mul(rem(div(lane, 8), 2), 4), mul(rem(lane, 2), 2));
  return {r, col};
}

/// The memref indices of the two fragment elements of the 8x8 block at
/// (row, col).
SmallVector<std::pair<Value, Value>, 2>
fragmentIndices(OpBuilder &b, Location loc, Value row, Value col,
                bool transpose) {
  auto [r, c] = fragmentPosition(b, loc);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value r0 = arith::AddIOp::create(b, loc, row, transpose ? c : r);
  Value c0 = arith::AddIOp::create(b, loc, col, transpose ? r : c);
  // The second element is (r, c + 1); transposed, its memref row moves.
  if (transpose)
    return {{r0, c0}, {arith::AddIOp::create(b, loc, r0, one), c0}};
  return {{r0, c0}, {r0, arith::AddIOp::create(b, loc, c0, one)}};
}

struct ExpandMatrixLoad : OpRewritePattern<MatrixLoadOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatrixLoadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 2> elements;
    for (auto [i, j] : fragmentIndices(rewriter, loc, op.getRow(), op.getCol(),
                                       op.getTranspose()))
      elements.push_back(
          memref::LoadOp::create(rewriter, loc, op.getBase(), ValueRange{i, j}));
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(op, op.getType(),
                                                        elements);
    return success();
  }
};

struct ExpandMatrixStore : OpRewritePattern<MatrixStoreOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MatrixStoreOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    for (auto [n, ij] : llvm::enumerate(fragmentIndices(
             rewriter, loc, op.getRow(), op.getCol(), op.getTranspose()))) {
      Value e = vector::ExtractOp::create(rewriter, loc, op.getValue(),
                                          ArrayRef<int64_t>{(int64_t)n});
      memref::StoreOp::create(rewriter, loc, e, op.getBase(),
                              ValueRange{ij.first, ij.second});
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Mixed precision and the negated accumulator become conversions and a
/// same-type agx.mma where the matrix unit does not do them.
struct ExpandMMA : OpRewritePattern<MMAOp> {
  ExpandMMA(MLIRContext *ctx, const ChipInfo &chip)
      : OpRewritePattern(ctx), chip(chip) {}
  LogicalResult matchAndRewrite(MMAOp op,
                                PatternRewriter &rewriter) const override {
    bool mixed = op.getA().getType() != op.getC().getType();
    bool negate = op.getNegateAcc() &&
                  !(chip.matrixUnit == MatrixUnit::Native &&
                    chip.matrixNegatedAccumulate);
    if (!mixed && !negate)
      return failure();
    Location loc = op.getLoc();
    Type accType = op.getC().getType();
    Value a = op.getA(), b = op.getB(), c = op.getC();
    if (mixed) {
      a = arith::ExtFOp::create(rewriter, loc, accType, a);
      b = arith::ExtFOp::create(rewriter, loc, accType, b);
    }
    if (negate)
      c = arith::NegFOp::create(rewriter, loc, c);
    auto mma = MMAOp::create(rewriter, loc, accType, a, b, c);
    mma.setNegateAcc(op.getNegateAcc() && !negate);
    rewriter.replaceOp(op, mma);
    return success();
  }
  ChipInfo chip;
};

/// The tile product as fragment loads and agx.mma, each A and B fragment
/// loaded once per step of k.
struct ExpandMMATile : OpRewritePattern<MMATileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(MMATileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t mt = op.getM() / 8, nt = op.getN() / 8, kt = op.getK() / 8;
    Type inElt = op.getA().getType().getElementType();
    Type accElt = op.getAcc().getType().getElementType();
    auto inFrag = VectorType::get({2}, inElt);
    auto accFrag = VectorType::get({2}, accElt);
    auto offset = [&](Value base, int64_t v) -> Value {
      if (v == 0)
        return base;
      return arith::AddIOp::create(rewriter, loc, base,
                                   arith::ConstantIndexOp::create(rewriter, loc, v));
    };
    SmallVector<Value> acc;
    for (int64_t f = 0; f < mt * nt; ++f)
      acc.push_back(vector::ExtractStridedSliceOp::create(
          rewriter, loc, op.getAcc(), ArrayRef<int64_t>{2 * f},
          ArrayRef<int64_t>{2}, ArrayRef<int64_t>{1}));
    for (int64_t k = 0; k < kt; ++k) {
      SmallVector<Value> as, bs;
      for (int64_t i = 0; i < mt; ++i)
        as.push_back(MatrixLoadOp::create(
            rewriter, loc, inFrag, op.getA(), offset(op.getARow(), 8 * i),
            offset(op.getACol(), 8 * k), /*transpose=*/false));
      for (int64_t j = 0; j < nt; ++j)
        bs.push_back(MatrixLoadOp::create(
            rewriter, loc, inFrag, op.getB(), offset(op.getBRow(), 8 * k),
            offset(op.getBCol(), 8 * j), /*transpose=*/false));
      for (int64_t i = 0; i < mt; ++i)
        for (int64_t j = 0; j < nt; ++j)
          acc[i * nt + j] = MMAOp::create(rewriter, loc, accFrag, as[i], bs[j],
                                          acc[i * nt + j]);
    }
    Value result = op.getAcc();
    for (int64_t f = 0; f < mt * nt; ++f)
      result = vector::InsertStridedSliceOp::create(
          rewriter, loc, acc[f], result, ArrayRef<int64_t>{2 * f},
          ArrayRef<int64_t>{1});
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// shuffle_up(a, shift) * b + c within segments: lanes below `shift` in
/// their segment, or whose source lane is inactive, read 0.
struct ExpandFMALaneShift : OpRewritePattern<FMALaneShiftOp> {
  ExpandFMALaneShift(MLIRContext *ctx, const ChipInfo &chip)
      : OpRewritePattern(ctx), chip(chip) {}
  LogicalResult matchAndRewrite(FMALaneShiftOp op,
                                PatternRewriter &rewriter) const override {
    if (chip.laneShiftFMA)
      return failure();
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    auto ci = [&](int64_t v) { return arith::ConstantIntOp::create(rewriter, loc, i32, v); };
    Value shift = ci(op.getShift());
    Value shifted = ShuffleOp::create(rewriter, loc, op.getA().getType(),
                                      ShuffleMode::up, op.getA(), shift,
                                      Scope::simdgroup);
    Value lane = arith::IndexCastUIOp::create(
        rewriter, loc, i32, gpu::LaneIdOp::create(rewriter, loc, nullptr));
    Value pos = arith::RemUIOp::create(rewriter, loc, lane, ci(op.getWidth()));
    Value inSegment = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::uge,
                                            pos, shift);
    // The source lane must be active: bit (lane - shift) of the ballot.
    Value active = BallotOp::create(rewriter, loc, i32,
                                    arith::ConstantIntOp::create(rewriter, loc, 1, 1),
                                    Scope::simdgroup);
    Value src = arith::SubIOp::create(rewriter, loc, lane, shift);
    Value bit = arith::AndIOp::create(
        rewriter, loc, arith::ShRUIOp::create(rewriter, loc, active, src), ci(1));
    Value srcActive = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                            bit, ci(0));
    Value valid = arith::AndIOp::create(rewriter, loc, inSegment, srcActive);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(op.getA().getType(), 0.0));
    Value a = arith::SelectOp::create(rewriter, loc, valid, shifted, zero);
    rewriter.replaceOpWithNewOp<math::FmaOp>(op, a, op.getB(), op.getC());
    return success();
  }
  ChipInfo chip;
};

struct ExpandBlockBarrier : OpRewritePattern<BlockBarrierOp> {
  ExpandBlockBarrier(MLIRContext *ctx, const ChipInfo &chip)
      : OpRewritePattern(ctx), chip(chip) {}
  LogicalResult matchAndRewrite(BlockBarrierOp op,
                                PatternRewriter &rewriter) const override {
    if (chip.blockBarrier)
      return failure();
    rewriter.replaceOpWithNewOp<ThreadgroupBarrierOp>(op);
    return success();
  }
  ChipInfo chip;
};

/// Until the runtime installs preambles, uniform regions run in every thread.
struct InlineUniformRegion : OpRewritePattern<UniformRegionOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(UniformRegionOp op,
                                PatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    SmallVector<Value> results(yield.getValues());
    rewriter.eraseOp(yield);
    rewriter.inlineBlockBefore(&body, op);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct AGXExpandOpsPass
    : public agx::impl::AGXExpandOpsPassBase<AGXExpandOpsPass> {
  using Base::Base;
  void runOnOperation() override {
    // Each gpu.module is expanded for its own chip.
    SmallVector<Operation *> roots;
    getOperation()->walk([&](gpu::GPUModuleOp m) { roots.push_back(m); });
    if (roots.empty())
      roots.push_back(getOperation());
    for (Operation *root : roots) {
      // getTargetChip looks at enclosing modules; a root module is its own.
      StringRef name = chip;
      StringRef targetFeatures = features;
      if (name.empty()) {
        name = getTargetChip(root);
        if (auto m = dyn_cast<gpu::GPUModuleOp>(root))
          if (ArrayAttr targets = m.getTargetsAttr())
            for (Attribute t : targets)
              if (auto target = dyn_cast<AGXIR::AGXIRTargetAttr>(t)) {
                name = target.getChip();
                if (targetFeatures.empty())
                  targetFeatures = target.getFeatures();
              }
      }
      FailureOr<ChipInfo> info = ChipInfo::get(name, targetFeatures);
      if (failed(info)) {
        root->emitError() << "unknown AGX chip '" << name << "'"
                          << (targetFeatures.contains("+mesa-isa")
                                  ? " for +mesa-isa (G13 and G14 only)"
                                  : "");
        return signalPassFailure();
      }
      RewritePatternSet patterns(&getContext());
      populateAGXExpandOpsPatterns(patterns, *info);
      if (failed(applyPatternsGreedily(root, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

void mlir::agx::populateAGXExpandOpsPatterns(RewritePatternSet &patterns,
                                             const ChipInfo &chip) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ExpandMatrixLoad, ExpandMatrixStore, ExpandMMATile,
               InlineUniformRegion>(ctx);
  patterns.add<ExpandMMA, ExpandFMALaneShift, ExpandBlockBarrier>(ctx, chip);
}
