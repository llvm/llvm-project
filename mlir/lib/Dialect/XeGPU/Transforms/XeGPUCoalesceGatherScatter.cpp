//===- XeGPUCoalesceGatherScatter.cpp - Coalesce scatter accesses --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass coalesces neighbouring lanes of `xegpu.load` / `xegpu.store` ops
// into accesses with a larger `chunk_size`. It targets the scatter-style
// pointer/memref + offsets form of these ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUCOALESCEGATHERSCATTER
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-coalesce-gather-scatter"

using namespace mlir;

namespace {

/// Description of an offsets vector that has been recognized as coalescible.
struct OffsetsPattern {
  enum class Kind {
    /// All lanes load the same address: rewrite as a broadcast.
    Broadcast,
    /// Lane `i` loads at `base + i * stride * elementSize` (in element units).
    Affine,
  };
  Kind kind;
  /// Stride in element units between adjacent lanes. Only meaningful for
  /// `Affine`.
  int64_t stride = 0;
};

/// Match a vector of `index` against the supported coalescible patterns.
///
/// Recognized:
///   - `arith.constant dense<C>` (all lanes equal)         -> Broadcast
///   - `arith.constant dense<[a, a+s, a+2s, ...]>`         -> Affine(s)
///   - `vector.step` (lane `i` -> i)                       -> Affine(1)
///   - `arith.muli %step, %splat<S>` / `arith.muli %splat<S>, %step`
///                                                          -> Affine(S)
///   - `arith.addi %x, %splat<base>` / commuted             -> recurse on %x
static std::optional<OffsetsPattern> matchOffsetsPattern(Value offsets) {
  auto vecTy = dyn_cast<VectorType>(offsets.getType());
  if (!vecTy || vecTy.getRank() != 1)
    return std::nullopt;
  // Length-1 offsets vectors carry no coalescing opportunity and would let
  // the broadcast-rewrite output re-match itself in the greedy driver.
  if (vecTy.getNumElements() <= 1)
    return std::nullopt;

  // arith.constant dense<...>
  if (auto cst = offsets.getDefiningOp<arith::ConstantOp>()) {
    auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue());
    if (!dense)
      return std::nullopt;
    if (dense.isSplat()) {
      OffsetsPattern p;
      p.kind = OffsetsPattern::Kind::Broadcast;
      return p;
    }
    // Check if values form an arithmetic progression.
    auto values = llvm::to_vector(dense.getValues<APInt>());
    if (values.size() < 2)
      return std::nullopt;
    int64_t stride =
        values[1].getSExtValue() - values[0].getSExtValue();
    for (size_t i = 2; i < values.size(); ++i) {
      int64_t diff = values[i].getSExtValue() - values[i - 1].getSExtValue();
      if (diff != stride)
        return std::nullopt;
    }
    OffsetsPattern p;
    if (stride == 0) {
      p.kind = OffsetsPattern::Kind::Broadcast;
    } else {
      p.kind = OffsetsPattern::Kind::Affine;
      p.stride = stride;
    }
    return p;
  }

  // vector.step -> stride 1.
  if (offsets.getDefiningOp<vector::StepOp>()) {
    OffsetsPattern p;
    p.kind = OffsetsPattern::Kind::Affine;
    p.stride = 1;
    return p;
  }

  // Helper to recognize a vector splat with constant integer value.
  auto matchIndexSplat = [](Value v) -> std::optional<int64_t> {
    if (auto bcast = v.getDefiningOp<vector::BroadcastOp>()) {
      APInt c;
      if (matchPattern(bcast.getSource(), m_ConstantInt(&c)))
        return c.getSExtValue();
    }
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue()))
        if (dense.isSplat())
          return dense.getSplatValue<APInt>().getSExtValue();
    }
    return std::nullopt;
  };

  // arith.muli with one splat operand.
  if (auto mul = offsets.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mul.getLhs(), rhs = mul.getRhs();
    auto lhsSplat = matchIndexSplat(lhs);
    auto rhsSplat = matchIndexSplat(rhs);
    Value nonSplat = lhsSplat ? rhs : (rhsSplat ? lhs : Value());
    std::optional<int64_t> factor = lhsSplat ? lhsSplat : rhsSplat;
    if (nonSplat && factor) {
      auto inner = matchOffsetsPattern(nonSplat);
      if (inner && inner->kind == OffsetsPattern::Kind::Affine) {
        OffsetsPattern p;
        p.kind = OffsetsPattern::Kind::Affine;
        p.stride = inner->stride * (*factor);
        return p;
      }
      if (inner && inner->kind == OffsetsPattern::Kind::Broadcast) {
        // splat * splat is still uniform.
        return inner;
      }
    }
  }

  // arith.addi with a splat operand: stride is unaffected.
  if (auto add = offsets.getDefiningOp<arith::AddIOp>()) {
    Value lhs = add.getLhs(), rhs = add.getRhs();
    auto lhsSplat = matchIndexSplat(lhs);
    auto rhsSplat = matchIndexSplat(rhs);
    Value nonSplat = lhsSplat ? rhs : (rhsSplat ? lhs : Value());
    if (nonSplat)
      return matchOffsetsPattern(nonSplat);
  }

  return std::nullopt;
}

/// Returns true if `mask` is a constant `dense<true>` vector.
static bool isAllTrueMask(Value mask) {
  auto vecTy = dyn_cast<VectorType>(mask.getType());
  if (!vecTy)
    return false;
  auto cst = mask.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat())
    return false;
  return dense.getSplatValue<APInt>().getBoolValue();
}

/// Compute the largest factor `N` such that:
///   - `N` divides `numLanes`,
///   - `N <= maxChunkSize / origChunkSize`,
///   - lane stride (in elements) equals 1 when scaled by N (i.e. `N * stride
///     == N * stride`, requires stride to be 1 for true contiguity at the
///     coalesced granularity).
///
/// Returns std::nullopt if no useful coalescing factor exists.
static std::optional<int64_t> chooseAffineCoalesceFactor(int64_t numLanes,
                                                         int64_t origChunk,
                                                         int64_t stride,
                                                         unsigned maxChunkSize) {
  if (stride != 1)
    return std::nullopt;
  if (numLanes < 2)
    return std::nullopt;
  if (origChunk < 1)
    origChunk = 1;
  int64_t budget = static_cast<int64_t>(maxChunkSize) / origChunk;
  if (budget < 2)
    return std::nullopt;
  // Pick the largest power-of-two factor that divides numLanes and fits
  // budget.
  int64_t factor = 1;
  for (int64_t f = std::min<int64_t>(budget, numLanes); f >= 2; f /= 2) {
    if (numLanes % f == 0) {
      factor = f;
      break;
    }
  }
  if (factor < 2)
    return std::nullopt;
  return factor;
}

/// Pick the new offsets value: a sub-vector of length `newLen` extracted from
/// the original offsets, taking every `factor`-th element. For affine offsets
/// generated from `vector.step` + scalar adds/muls, the original offsets
/// already encode the right values; the simplest valid construction is to
/// rebuild them using the same defining chain at the smaller length.
///
/// We take a pragmatic approach: emit
///   %step  = vector.step : vector<newLen x index>
///   %scaled = arith.muli %step, %splat<factor*stride_element>
///   %final  = arith.addi %scaled, %splat<lane0_offset>
///
/// where `lane0_offset` is `offsets[0]` extracted at runtime via
/// `vector.extract`. For dense-constant offsets we just emit a smaller dense
/// constant directly.
static Value buildCoalescedAffineOffsets(OpBuilder &b, Location loc,
                                         Value origOffsets, int64_t newLen,
                                         int64_t factor) {
  auto idxTy = b.getIndexType();
  auto newVecTy = VectorType::get({newLen}, idxTy);

  // Fast path: dense constant offsets -> emit a smaller dense constant.
  if (auto cst = origOffsets.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      auto srcVals = llvm::to_vector(dense.getValues<APInt>());
      SmallVector<APInt> newVals;
      newVals.reserve(newLen);
      for (int64_t i = 0; i < newLen; ++i)
        newVals.push_back(srcVals[i * factor]);
      auto newAttr =
          DenseIntElementsAttr::get(newVecTy, ArrayRef<APInt>(newVals));
      return arith::ConstantOp::create(b, loc, newAttr);
    }
  }

  // General path: rebuild using vector.step + scalar broadcast/muli/addi.
  // Lane-0 offset is the first element of the original offsets.
  Value zero = arith::ConstantIndexOp::create(b, loc, 0);
  Value lane0 =
      vector::ExtractOp::create(b, loc, origOffsets, ArrayRef<int64_t>{0});
  Value step = vector::StepOp::create(b, loc, newVecTy);
  Value factorSplat = arith::ConstantOp::create(
      b, loc,
      DenseIntElementsAttr::get(newVecTy,
                                APInt(64, factor, /*isSigned=*/true)));
  // Note: vector.step yields index, factorSplat must also be index-typed.
  // Build factor splat as index using broadcast of scalar.
  factorSplat = vector::BroadcastOp::create(
      b, loc, newVecTy,
      arith::ConstantIndexOp::create(b, loc, factor).getResult());
  Value scaled = arith::MulIOp::create(b, loc, step, factorSplat);
  Value baseSplat = vector::BroadcastOp::create(b, loc, newVecTy, lane0);
  Value result = arith::AddIOp::create(b, loc, scaled, baseSplat);
  (void)zero;
  return result;
}

/// Build a `dense<true>` mask vector of length `newLen`.
static Value buildAllTrueMask(OpBuilder &b, Location loc, int64_t newLen) {
  auto vecTy = VectorType::get({newLen}, b.getI1Type());
  auto attr = DenseIntElementsAttr::get(vecTy, true);
  return arith::ConstantOp::create(b, loc, attr);
}

/// Pattern that coalesces a `xegpu.load` op.
struct CoalesceLoadPattern final : OpRewritePattern<xegpu::LoadGatherOp> {
  CoalesceLoadPattern(MLIRContext *ctx, unsigned maxChunkSize)
      : OpRewritePattern(ctx), maxChunkSize(maxChunkSize) {}

  LogicalResult matchAndRewrite(xegpu::LoadGatherOp op,
                                PatternRewriter &rewriter) const override {
    // Only the source-as-pointer/memref form has a vector offsets operand we
    // can analyze; the tensor_desc form has no offsets here.
    auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
    if (!offsetsTy || offsetsTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D vector offsets");
    if (offsetsTy.getNumElements() <= 1)
      return rewriter.notifyMatchFailure(op, "nothing to coalesce");
    auto valueTy = op.getValueType();
    if (!valueTy || valueTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D vector value");
    if (!isAllTrueMask(op.getMask()))
      return rewriter.notifyMatchFailure(op, "non-uniform mask");

    auto pattern = matchOffsetsPattern(op.getOffsets());
    if (!pattern)
      return rewriter.notifyMatchFailure(op, "offsets not coalescible");

    int64_t numLanes = offsetsTy.getNumElements();
    int64_t origChunk =
        static_cast<int64_t>(op.getChunkSize().value_or(1));

    Location loc = op.getLoc();

    if (pattern->kind == OffsetsPattern::Kind::Broadcast) {
      // All lanes load the same address: emit a single scalar load and
      // broadcast.
      Value scalarOffset = vector::ExtractOp::create(
          rewriter, loc, op.getOffsets(), ArrayRef<int64_t>{0});
      Value scalarMask = arith::ConstantOp::create(
          rewriter, loc, rewriter.getOneAttr(rewriter.getI1Type()));
      // Result of the scalar load matches the original element type with the
      // chunk dimension preserved if any (1-D value, length = chunk_size or 1).
      int64_t scalarLen = numLanes; // value length per lane is implicit; for
                                     // 1-D the whole vector represents lanes.
      // Build a length-1 offsets vector to keep types consistent (the op
      // accepts scalar offsets too, but our incoming form is vector).
      auto idxVecTy = VectorType::get({1}, rewriter.getIndexType());
      auto maskVecTy = VectorType::get({1}, rewriter.getI1Type());
      Value newOffsets = vector::BroadcastOp::create(
          rewriter, loc, idxVecTy, scalarOffset);
      Value newMask = arith::ConstantOp::create(
          rewriter, loc, DenseIntElementsAttr::get(maskVecTy, true));
      auto newValueTy =
          VectorType::get({1}, valueTy.getElementType());
      auto newLoad = xegpu::LoadGatherOp::create(
          rewriter, loc, newValueTy, op.getSource(), newOffsets, newMask,
          /*chunk_size=*/IntegerAttr(), op.getL1HintAttr(),
          op.getL2HintAttr(), op.getL3HintAttr(),
          /*layout=*/xegpu::DistributeLayoutAttr());
      // Splat to original shape.
      Value scalar = vector::ExtractOp::create(
          rewriter, loc, newLoad.getResult(), ArrayRef<int64_t>{0});
      Value bcast =
          vector::BroadcastOp::create(rewriter, loc, valueTy, scalar);
      rewriter.replaceOp(op, bcast);
      (void)scalarMask;
      (void)scalarLen;
      return success();
    }

    // Affine path.
    auto factorOpt = chooseAffineCoalesceFactor(numLanes, origChunk,
                                                pattern->stride, maxChunkSize);
    if (!factorOpt)
      return rewriter.notifyMatchFailure(op, "no useful coalesce factor");
    int64_t factor = *factorOpt;
    int64_t newLanes = numLanes / factor;
    int64_t newChunk = origChunk * factor;

    Value newOffsets = buildCoalescedAffineOffsets(rewriter, loc,
                                                   op.getOffsets(),
                                                   newLanes, factor);
    Value newMask = buildAllTrueMask(rewriter, loc, newLanes);

    // The verifier requires `chunk_size > 1` to be paired with a 2-D value
    // vector of shape `<lanes x chunk>`. Build that 2-D type and shape_cast
    // back to the original 1-D shape for the consumer.
    SmallVector<int64_t> newShape;
    if (valueTy.getRank() == 1) {
      newShape = {newLanes, newChunk};
    } else {
      newShape = llvm::to_vector(valueTy.getShape());
      newShape[0] = newLanes;
      newShape.back() = newChunk;
    }
    auto newValueTy = VectorType::get(newShape, valueTy.getElementType());

    auto newChunkAttr = rewriter.getI64IntegerAttr(newChunk);
    auto newLoad = xegpu::LoadGatherOp::create(
        rewriter, loc, newValueTy, op.getSource(), newOffsets, newMask,
        newChunkAttr, op.getL1HintAttr(), op.getL2HintAttr(),
        op.getL3HintAttr(), /*layout=*/xegpu::DistributeLayoutAttr());

    if (newValueTy == valueTy) {
      rewriter.replaceOp(op, newLoad.getResult());
    } else {
      Value reshaped = vector::ShapeCastOp::create(
          rewriter, loc, valueTy, newLoad.getResult());
      rewriter.replaceOp(op, reshaped);
    }
    return success();
  }

  unsigned maxChunkSize;
};

/// Pattern that coalesces a `xegpu.store` op.
struct CoalesceStorePattern final : OpRewritePattern<xegpu::StoreScatterOp> {
  CoalesceStorePattern(MLIRContext *ctx, unsigned maxChunkSize)
      : OpRewritePattern(ctx), maxChunkSize(maxChunkSize) {}

  LogicalResult matchAndRewrite(xegpu::StoreScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto offsetsTy = dyn_cast<VectorType>(op.getOffsets().getType());
    if (!offsetsTy || offsetsTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D vector offsets");
    if (offsetsTy.getNumElements() <= 1)
      return rewriter.notifyMatchFailure(op, "nothing to coalesce");
    auto valueTy = op.getValueType();
    if (!valueTy || valueTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1-D vector value");
    if (!isAllTrueMask(op.getMask()))
      return rewriter.notifyMatchFailure(op, "non-uniform mask");

    auto pattern = matchOffsetsPattern(op.getOffsets());
    if (!pattern)
      return rewriter.notifyMatchFailure(op, "offsets not coalescible");

    // For the broadcast case, all lanes write the same address. That is not a
    // meaningful coalesce target (last writer wins, semantics-preserving
    // rewrite would need a single arbitrary lane to win). Skip.
    if (pattern->kind == OffsetsPattern::Kind::Broadcast)
      return rewriter.notifyMatchFailure(
          op, "all-equal offsets on store would be ambiguous");

    int64_t numLanes = offsetsTy.getNumElements();
    int64_t origChunk =
        static_cast<int64_t>(op.getChunkSize().value_or(1));
    auto factorOpt = chooseAffineCoalesceFactor(numLanes, origChunk,
                                                pattern->stride, maxChunkSize);
    if (!factorOpt)
      return rewriter.notifyMatchFailure(op, "no useful coalesce factor");
    int64_t factor = *factorOpt;
    int64_t newLanes = numLanes / factor;
    int64_t newChunk = origChunk * factor;

    Location loc = op.getLoc();
    Value newOffsets = buildCoalescedAffineOffsets(rewriter, loc,
                                                   op.getOffsets(),
                                                   newLanes, factor);
    Value newMask = buildAllTrueMask(rewriter, loc, newLanes);

    auto newChunkAttr = rewriter.getI64IntegerAttr(newChunk);
    // The verifier requires `chunk_size > 1` to be paired with a 2-D value
    // vector of shape `<lanes x chunk>`; shape_cast the original 1-D value
    // into that shape.
    SmallVector<int64_t> newShape;
    if (valueTy.getRank() == 1)
      newShape = {newLanes, newChunk};
    else {
      newShape = llvm::to_vector(valueTy.getShape());
      newShape[0] = newLanes;
      newShape.back() = newChunk;
    }
    auto newValTy = VectorType::get(newShape, valueTy.getElementType());
    Value newValue = op.getValue();
    if (newValTy != valueTy)
      newValue =
          vector::ShapeCastOp::create(rewriter, loc, newValTy, op.getValue());
    xegpu::StoreScatterOp::create(rewriter, loc, newValue, op.getDest(),
                                  newOffsets, newMask, newChunkAttr,
                                  op.getL1HintAttr(), op.getL2HintAttr(),
                                  op.getL3HintAttr(),
                                  /*layout=*/xegpu::DistributeLayoutAttr());
    rewriter.eraseOp(op);
    return success();
  }

  unsigned maxChunkSize;
};

struct XeGPUCoalesceGatherScatterPass final
    : public xegpu::impl::XeGPUCoalesceGatherScatterBase<
          XeGPUCoalesceGatherScatterPass> {
  using XeGPUCoalesceGatherScatterBase::XeGPUCoalesceGatherScatterBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CoalesceLoadPattern, CoalesceStorePattern>(ctx, maxChunkSize);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
