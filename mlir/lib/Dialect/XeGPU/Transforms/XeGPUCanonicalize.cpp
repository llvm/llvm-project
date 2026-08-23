//===- XeGPUCanonicalize.cpp - XeGPU specific canonicalization --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/Dialect/XeGPU/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace xegpu {
#define GEN_PASS_DEF_XEGPUCANONICALIZE
#include "mlir/Dialect/XeGPU/Transforms/Passes.h.inc"
} // namespace xegpu
} // namespace mlir

#define DEBUG_TYPE "xegpu-canonicalize"

using namespace mlir;

namespace {

/// Returns the `vector.shape_cast` that flattened a value of type `ndType`
/// into `flat`, if that is how `flat` was produced.
static vector::ShapeCastOp getFlattenCast(Value flat, VectorType ndType) {
  auto shapeCast = flat.getDefiningOp<vector::ShapeCastOp>();
  if (shapeCast && shapeCast.getSourceVectorType() == ndType)
    return shapeCast;
  return nullptr;
}

static DenseElementsAttr getDenseConstant(Value flat) {
  DenseElementsAttr elements;
  if (matchPattern(flat, m_Constant(&elements)))
    return elements;
  return nullptr;
}

static vector::BroadcastOp getSplatBroadcast(Value flat) {
  auto broadcast = flat.getDefiningOp<vector::BroadcastOp>();
  if (broadcast && !isa<VectorType>(broadcast.getSourceType()))
    return broadcast;
  return nullptr;
}

/// Returns true if `unflatten` can reshape `flat` to `ndType`.
static bool canUnflatten(Value flat, VectorType ndType) {
  return getFlattenCast(flat, ndType) || getDenseConstant(flat) ||
         getSplatBroadcast(flat);
}

/// Reshape `flat` to `ndType` without introducing a 1-D to N-D
/// `vector.shape_cast`. Only handles the forms a frontend actually emits when
/// flattening: the flattening cast itself, a constant, and a splat.
/// `canUnflatten` must hold.
static Value unflatten(PatternRewriter &rewriter, Value flat,
                       VectorType ndType) {
  assert(cast<VectorType>(flat.getType()).getRank() == 1 &&
         "expected a 1-D vector");

  if (auto shapeCast = getFlattenCast(flat, ndType))
    return shapeCast.getSource();

  if (DenseElementsAttr elements = getDenseConstant(flat))
    return arith::ConstantOp::create(rewriter, flat.getLoc(), ndType,
                                     elements.reshape(ndType));

  auto broadcast = getSplatBroadcast(flat);
  assert(broadcast && "expected canUnflatten to hold");
  return vector::BroadcastOp::create(rewriter, flat.getLoc(), ndType,
                                     broadcast.getSource());
}

/// Restore the N-D form of a flattened `vector.gather` / `vector.scatter`.
///
/// XeGPU layouts are expressed in terms of the N-D shape of the accessed data,
/// so a flattened gather/scatter forces layout propagation to reason through
/// the surrounding `vector.shape_cast` ops - which is either impossible or
/// yields layouts that cannot be distributed.
///
/// ```mlir
/// // Before:
/// %cst = arith.constant dense<0.0> : vector<8192xbf16>
/// %flat_idx = vector.shape_cast %idx : vector<128x64xindex> to vector<8192xindex>
/// %flat_mask = vector.shape_cast %mask : vector<128x64xi1> to vector<8192xi1>
/// %flat_res = vector.gather %src[%c0] [%flat_idx], %flat_mask, %cst
///     : memref<?xbf16>, vector<8192xindex>, vector<8192xi1>, vector<8192xbf16>
///       into vector<8192xbf16>
/// %res = vector.shape_cast %flat_res : vector<8192xbf16> to vector<128x64xbf16>
///
/// // After:
/// %cst = arith.constant dense<0.0> : vector<128x64xbf16>
/// %res = vector.gather %src[%c0] [%idx], %mask, %cst
///     : memref<?xbf16>, vector<128x64xindex>, vector<128x64xi1>,
///       vector<128x64xbf16> into vector<128x64xbf16>
/// ```
template <typename OpTy>
struct UnflattenGatherScatter : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    constexpr bool isGather = std::is_same_v<OpTy, vector::GatherOp>;

    if (op.getIndexVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(op, "index vector is not 1-D");

    // The N-D shape comes from the index operand's producer: this only undoes
    // a flattening that already happened, it never invents a shape.
    auto indexCast =
        op.getIndices().template getDefiningOp<vector::ShapeCastOp>();
    if (!indexCast || indexCast.getSourceVectorType().getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "index vector is not a shape_cast of an N-D vector");
    VectorType ndIndexType = indexCast.getSourceVectorType();
    VectorType ndMaskType =
        ndIndexType.cloneWith(std::nullopt, rewriter.getI1Type());
    VectorType ndType = ndIndexType.cloneWith(
        std::nullopt, op.getVectorType().getElementType());

    // Check everything before creating any IR: a partially applied rewrite
    // would leave dead ops behind.
    if (!canUnflatten(op.getMask(), ndMaskType))
      return rewriter.notifyMatchFailure(op, "cannot un-flatten the mask");

    if constexpr (isGather) {
      if (!canUnflatten(op.getPassThru(), ndType))
        return rewriter.notifyMatchFailure(op,
                                           "cannot un-flatten the pass-thru");

      // Every use must cast back to N-D, else the rewrite would just move the
      // shape_casts to the result.
      SmallVector<vector::ShapeCastOp> resultCasts;
      for (Operation *user : op->getUsers()) {
        auto resultCast = dyn_cast<vector::ShapeCastOp>(user);
        if (!resultCast || resultCast.getResultVectorType() != ndType)
          return rewriter.notifyMatchFailure(
              op, "result is not exclusively shape_cast back to N-D");
        resultCasts.push_back(resultCast);
      }
      if (resultCasts.empty())
        return rewriter.notifyMatchFailure(op, "result is unused");

      Value mask = unflatten(rewriter, op.getMask(), ndMaskType);
      Value passThru = unflatten(rewriter, op.getPassThru(), ndType);
      auto ndGather = vector::GatherOp::create(
          rewriter, op.getLoc(), ndType, op.getBase(), op.getOffsets(),
          indexCast.getSource(), mask, passThru, op.getAlignmentAttr());
      for (vector::ShapeCastOp resultCast : resultCasts)
        rewriter.replaceOp(resultCast, ndGather.getResult());
      rewriter.eraseOp(op);
    } else {
      if (!canUnflatten(op.getValueToStore(), ndType))
        return rewriter.notifyMatchFailure(
            op, "cannot un-flatten the stored value");

      Value mask = unflatten(rewriter, op.getMask(), ndMaskType);
      Value valueToStore = unflatten(rewriter, op.getValueToStore(), ndType);
      // Only operand types change, and this keeps the optional tensor result
      // untouched.
      rewriter.modifyOpInPlace(op, [&] {
        op.getIndicesMutable().assign(indexCast.getSource());
        op.getMaskMutable().assign(mask);
        op.getValueToStoreMutable().assign(valueToStore);
      });
    }
    return success();
  }
};

struct XeGPUCanonicalizePass final
    : public xegpu::impl::XeGPUCanonicalizeBase<XeGPUCanonicalizePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    xegpu::populateXeGPUCanonicalizePatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

void xegpu::populateXeGPUCanonicalizePatterns(RewritePatternSet &patterns) {
  patterns.add<UnflattenGatherScatter<vector::GatherOp>,
               UnflattenGatherScatter<vector::ScatterOp>>(
      patterns.getContext());
}
