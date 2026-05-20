//===- OnlineSoftmax.cpp - Rewrite softmax+matmul to online softmax -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pattern that recognizes softmax -> matmul and rewrites
// it into local_softmax + rescaling matmul (linalg.generic), enabling online
// (FlashAttention-style) computation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "linalg-online-softmax"

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Find a matmul user of the softmax result where:
/// - The softmax result is the LHS (input 0) of the matmul
/// - The softmax dimension matches the matmul contraction dimension
static linalg::MatmulOp findMatchingMatmulUser(linalg::SoftmaxOp softmaxOp) {
  Value softmaxResult = softmaxOp.getResult()[0];
  int64_t softmaxDim = softmaxOp.getDimension();

  for (Operation *user : softmaxResult.getUsers()) {
    auto matmulOp = dyn_cast<linalg::MatmulOp>(user);
    if (!matmulOp)
      continue;

    // Check that softmax result is the LHS (first input) of the matmul.
    if (matmulOp.getInputs()[0] != softmaxResult)
      continue;

    // For a standard matmul with inputs [M, K] x [K, N] -> [M, N],
    // the contraction dimension is 1 (the last dim of LHS).
    // The softmax dimension should match this contraction dim.
    auto lhsType = cast<RankedTensorType>(softmaxResult.getType());
    int64_t lhsRank = lhsType.getRank();
    // For standard matmul, contraction dim is the last dim of LHS.
    int64_t contractionDim = lhsRank - 1;

    if (softmaxDim == contractionDim)
      return matmulOp;
  }
  return nullptr;
}

/// Build the rescaling body (shared between rescaling matmul and rescaling
/// softmax). The body implements the online softmax correction algorithm.
///
/// Args layout: [p_val, m_tile, l_tile, v_val, O_acc, M_acc, L_acc]
static void buildRescalingBody(OpBuilder &b, Location loc, ValueRange args) {
  Value p_val = args[0], m_tile = args[1], l_tile = args[2], v_val = args[3];
  Value O_acc = args[4], M_acc = args[5], L_acc = args[6];

  // Step 1: M_new = max(M_acc, m_tile)
  Value M_new = arith::MaximumFOp::create(b, loc, M_acc, m_tile);

  // Step 2: Update L
  Value diff1 = arith::SubFOp::create(b, loc, M_acc, M_new);
  Value correction = math::ExpOp::create(b, loc, diff1);
  Value L_rescaled = arith::MulFOp::create(b, loc, L_acc, correction);
  Value diff2 = arith::SubFOp::create(b, loc, m_tile, M_new);
  Value exp_diff = math::ExpOp::create(b, loc, diff2);
  Value unnorm = arith::MulFOp::create(b, loc, p_val, l_tile);
  Value shifted = arith::MulFOp::create(b, loc, unnorm, exp_diff);
  Value L_new = arith::AddFOp::create(b, loc, L_rescaled, shifted);

  // Step 3: Rescale O
  Value scale = arith::DivFOp::create(b, loc, L_rescaled, L_new);
  Value O_rescaled = arith::MulFOp::create(b, loc, O_acc, scale);

  // Step 4: Accumulate contribution
  Value contrib = arith::MulFOp::create(b, loc, shifted, v_val);
  Value contrib_norm = arith::DivFOp::create(b, loc, contrib, L_new);
  Value O_new = arith::AddFOp::create(b, loc, O_rescaled, contrib_norm);

  linalg::YieldOp::create(b, loc, ValueRange{O_new, M_new, L_new});
}

/// Create a filled tensor (empty + fill).
static Value createFilledTensor(OpBuilder &b, Location loc,
                                ArrayRef<int64_t> shape, Type elementType,
                                Value fillValue) {
  Value empty =
      tensor::EmptyOp::create(b, loc, shape, elementType).getResult();
  return linalg::FillOp::create(b, loc, fillValue, empty).getResult(0);
}

/// Pattern: SoftmaxOp whose result feeds into a MatmulOp as LHS.
/// Rewrites to: local_softmax + rescaling_matmul generic.
/// If the softmax has other users besides the matched matmul, also emits
/// a rescaling_softmax generic to recover global softmax.
struct SoftmaxMatmulToOnlineSoftmax
    : public OpRewritePattern<linalg::SoftmaxOp> {
  SoftmaxMatmulToOnlineSoftmax(MLIRContext *ctx, int64_t tileSize)
      : OpRewritePattern(ctx), tileSize(tileSize) {}

  LogicalResult matchAndRewrite(linalg::SoftmaxOp softmaxOp,
                                PatternRewriter &rewriter) const override {
    // --- Match ---

    // Only tensor semantics supported.
    Value softmaxInput = softmaxOp.getInput();
    auto inputType = dyn_cast<RankedTensorType>(softmaxInput.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(softmaxOp, "input is not a tensor");

    // Must have tensor results (not memref).
    if (softmaxOp.getResult().empty())
      return rewriter.notifyMatchFailure(softmaxOp, "no tensor result");

    // Find a matching matmul user.
    linalg::MatmulOp matmulOp = findMatchingMatmulUser(softmaxOp);
    if (!matmulOp)
      return rewriter.notifyMatchFailure(
          softmaxOp, "no matmul user with matching contraction dim");

    int64_t softmaxDim = softmaxOp.getDimension();
    int64_t N = inputType.getShape()[softmaxDim];

    // Require static shape and divisibility.
    if (ShapedType::isDynamic(N))
      return rewriter.notifyMatchFailure(softmaxOp,
                                         "softmax dim is dynamic");
    if (N % tileSize != 0)
      return rewriter.notifyMatchFailure(
          softmaxOp, "softmax dim not divisible by tile size");

    int64_t tn = N / tileSize;
    int64_t ts = tileSize;

    // Get shapes. For the standard case: input is [M, N], V is [N, Kv].
    auto softmaxResultType =
        cast<RankedTensorType>(softmaxOp.getResult()[0].getType());
    int64_t inputRank = inputType.getRank();

    // We handle the 2D case: input [M, N], softmax dim = 1.
    // M is all dims except the softmax dim.
    if (inputRank != 2)
      return rewriter.notifyMatchFailure(softmaxOp,
                                         "only rank-2 inputs supported");
    if (softmaxDim != 1)
      return rewriter.notifyMatchFailure(softmaxOp,
                                         "only dimension(1) supported");

    int64_t M = inputType.getShape()[0];
    if (ShapedType::isDynamic(M))
      return rewriter.notifyMatchFailure(softmaxOp, "M dim is dynamic");

    // Get V (RHS of the matmul) and its shape.
    Value V = matmulOp.getInputs()[1];
    auto vType = cast<RankedTensorType>(V.getType());
    // V is [N, Kv] for standard matmul.
    if (vType.getRank() != 2)
      return rewriter.notifyMatchFailure(matmulOp, "V is not rank-2");
    int64_t Kv = vType.getShape()[1];
    if (ShapedType::isDynamic(Kv))
      return rewriter.notifyMatchFailure(matmulOp, "Kv dim is dynamic");

    Type elemType = inputType.getElementType();
    Location loc = softmaxOp.getLoc();

    // --- Rewrite ---

    // (a) Create empty tensors for local_softmax outputs:
    //     P: [M, tn, ts], m: [M, tn], l: [M, tn]
    Value P_init = tensor::EmptyOp::create(rewriter, loc,
                                           ArrayRef<int64_t>{M, tn, ts},
                                           elemType)
                       .getResult();
    Value m_init = tensor::EmptyOp::create(rewriter, loc,
                                           ArrayRef<int64_t>{M, tn}, elemType)
                       .getResult();
    Value l_init = tensor::EmptyOp::create(rewriter, loc,
                                           ArrayRef<int64_t>{M, tn}, elemType)
                       .getResult();

    // (b) Create linalg.local_softmax.
    auto localSoftmaxOp = linalg::LocalSoftmaxOp::create(
        rewriter, loc,
        /*resultTypes=*/
        TypeRange{RankedTensorType::get({M, tn, ts}, elemType),
                  RankedTensorType::get({M, tn}, elemType),
                  RankedTensorType::get({M, tn}, elemType)},
        /*input=*/softmaxInput,
        /*output=*/P_init,
        /*max=*/m_init,
        /*den=*/l_init,
        /*dimension=*/rewriter.getI64IntegerAttr(softmaxDim),
        /*tile_size=*/rewriter.getI64IntegerAttr(ts));

    Value P = localSoftmaxOp.getResults()[0];
    Value m = localSoftmaxOp.getResults()[1];
    Value l = localSoftmaxOp.getResults()[2];

    // (c) Reshape V: [N, Kv] -> [tn, ts, Kv]
    auto expandedVType = RankedTensorType::get({tn, ts, Kv}, elemType);
    SmallVector<ReassociationIndices> vReassoc = {{0, 1}, {2}};
    Value V_tiled =
        tensor::ExpandShapeOp::create(rewriter, loc, expandedVType, V, vReassoc);

    // (d) Create init tensors for rescaling matmul:
    //     O: [M, Kv] filled with 0.0
    //     M_run: [M, Kv] filled with -inf
    //     L_run: [M, Kv] filled with 0.0
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));
    Value negInf = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(
            elemType, APFloat::getInf(
                          cast<FloatType>(elemType).getFloatSemantics(), true)));

    Value O_init =
        createFilledTensor(rewriter, loc, {M, Kv}, elemType, zero);
    Value M_init =
        createFilledTensor(rewriter, loc, {M, Kv}, elemType, negInf);
    Value L_init =
        createFilledTensor(rewriter, loc, {M, Kv}, elemType, zero);

    // (e) Build the rescaling matmul linalg.generic.
    // Dimensions: (m, tn, ts, kv)
    //   m  = parallel, tn = reduction, ts = reduction, kv = parallel
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr d0, d1, d2, d3;
    bindDims(ctx, d0, d1, d2, d3);

    // Indexing maps for rescaling matmul:
    // P:   (m, tn, ts, kv) -> (m, tn, ts)
    // m:   (m, tn, ts, kv) -> (m, tn)
    // l:   (m, tn, ts, kv) -> (m, tn)
    // V:   (m, tn, ts, kv) -> (tn, ts, kv)
    // O:   (m, tn, ts, kv) -> (m, kv)
    // M:   (m, tn, ts, kv) -> (m, kv)
    // L:   (m, tn, ts, kv) -> (m, kv)
    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(4, 0, {d0, d1, d2}, ctx), // P
        AffineMap::get(4, 0, {d0, d1}, ctx),     // m
        AffineMap::get(4, 0, {d0, d1}, ctx),     // l
        AffineMap::get(4, 0, {d1, d2, d3}, ctx), // V
        AffineMap::get(4, 0, {d0, d3}, ctx),     // O
        AffineMap::get(4, 0, {d0, d3}, ctx),     // M
        AffineMap::get(4, 0, {d0, d3}, ctx),     // L
    };

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,  // m
        utils::IteratorType::reduction, // tn
        utils::IteratorType::reduction, // ts
        utils::IteratorType::parallel,  // kv
    };

    auto rescalingMatmulOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/
        TypeRange{RankedTensorType::get({M, Kv}, elemType),
                  RankedTensorType::get({M, Kv}, elemType),
                  RankedTensorType::get({M, Kv}, elemType)},
        /*inputs=*/ValueRange{P, m, l, V_tiled},
        /*outputs=*/ValueRange{O_init, M_init, L_init}, indexingMaps,
        iteratorTypes, buildRescalingBody);

    Value rescaledO = rescalingMatmulOp.getResult(0);

    // (f) Handle softmax result replacement.
    // Check if softmax has users other than the matched matmul.
    Value softmaxResult = softmaxOp.getResult()[0];
    bool hasOtherUsers = false;
    for (Operation *user : softmaxResult.getUsers()) {
      if (user != matmulOp) {
        hasOtherUsers = true;
        break;
      }
    }

    if (hasOtherUsers) {
      // Build the rescaling softmax generic to recover global softmax.
      // Uses identity matrix I_tiled: [tn, ts, N]
      // Dimensions: (m, tn, ts, n_s)
      //   m = parallel, tn = reduction, ts = reduction, n_s = parallel

      // Create identity tensor: I[N, N] then expand to [tn, ts, N].
      // For simplicity, use a linalg.generic that produces identity elements
      // using linalg.index ops.
      Value I_empty =
          tensor::EmptyOp::create(rewriter, loc,
                                  ArrayRef<int64_t>{tn, ts, N}, elemType)
              .getResult();

      Value one = arith::ConstantOp::create(
          rewriter, loc, rewriter.getFloatAttr(elemType, 1.0));

      // Build identity tensor with a generic using index ops.
      // I_tiled[t, s, n] = 1.0 if t*ts + s == n, else 0.0
      AffineExpr i0, i1, i2;
      bindDims(ctx, i0, i1, i2);
      SmallVector<AffineMap> identityMaps = {
          AffineMap::get(3, 0, {i0, i1, i2}, ctx), // output
      };
      SmallVector<utils::IteratorType> identityIters = {
          utils::IteratorType::parallel,
          utils::IteratorType::parallel,
          utils::IteratorType::parallel,
      };

      auto identityGeneric = linalg::GenericOp::create(
          rewriter, loc,
          TypeRange{RankedTensorType::get({tn, ts, N}, elemType)},
          /*inputs=*/ValueRange{},
          /*outputs=*/ValueRange{I_empty}, identityMaps, identityIters,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            // I_tiled[t, s, n] = 1.0 if t*ts + s == n, else 0.0
            Value tIdx = linalg::IndexOp::create(b, nestedLoc, 0);
            Value sIdx = linalg::IndexOp::create(b, nestedLoc, 1);
            Value nIdx = linalg::IndexOp::create(b, nestedLoc, 2);
            Value tsConst = arith::ConstantIndexOp::create(b, nestedLoc, ts);
            Value tTimesTs = arith::MulIOp::create(b, nestedLoc, tIdx, tsConst);
            Value globalIdx =
                arith::AddIOp::create(b, nestedLoc, tTimesTs, sIdx);
            Value cond = arith::CmpIOp::create(b, nestedLoc,
                                               arith::CmpIPredicate::eq,
                                               globalIdx, nIdx);
            Value oneVal = arith::ConstantOp::create(
                b, nestedLoc, b.getFloatAttr(elemType, 1.0));
            Value zeroVal = arith::ConstantOp::create(
                b, nestedLoc, b.getFloatAttr(elemType, 0.0));
            Value result =
                arith::SelectOp::create(b, nestedLoc, cond, oneVal, zeroVal);
            linalg::YieldOp::create(b, nestedLoc, result);
          });

      Value I_tiled = identityGeneric.getResult(0);

      // Init tensors for rescaling softmax: O_s:[M, N], M_s:[M, N], L_s:[M, N]
      Value Os_init =
          createFilledTensor(rewriter, loc, {M, N}, elemType, zero);
      Value Ms_init =
          createFilledTensor(rewriter, loc, {M, N}, elemType, negInf);
      Value Ls_init =
          createFilledTensor(rewriter, loc, {M, N}, elemType, zero);

      // Indexing maps for rescaling softmax (dims: m, tn, ts, n_s):
      SmallVector<AffineMap> softmaxMaps = {
          AffineMap::get(4, 0, {d0, d1, d2}, ctx), // P
          AffineMap::get(4, 0, {d0, d1}, ctx),     // m
          AffineMap::get(4, 0, {d0, d1}, ctx),     // l
          AffineMap::get(4, 0, {d1, d2, d3}, ctx), // I_tiled
          AffineMap::get(4, 0, {d0, d3}, ctx),     // O_s
          AffineMap::get(4, 0, {d0, d3}, ctx),     // M_s
          AffineMap::get(4, 0, {d0, d3}, ctx),     // L_s
      };

      auto rescalingSoftmaxOp = linalg::GenericOp::create(
          rewriter, loc,
          TypeRange{RankedTensorType::get({M, N}, elemType),
                    RankedTensorType::get({M, N}, elemType),
                    RankedTensorType::get({M, N}, elemType)},
          /*inputs=*/ValueRange{P, m, l, I_tiled},
          /*outputs=*/ValueRange{Os_init, Ms_init, Ls_init}, softmaxMaps,
          iteratorTypes, buildRescalingBody);

      Value recoveredSoftmax = rescalingSoftmaxOp.getResult(0);

      // Replace all uses of the original softmax result (except the matmul)
      // with the recovered softmax.
      rewriter.replaceAllUsesExcept(softmaxResult, recoveredSoftmax, matmulOp);
    }

    // (g) Replace the matmul result with rescaledO.
    rewriter.replaceOp(matmulOp, rescaledO);

    // (h) If the softmax now has no remaining users, erase it.
    if (softmaxResult.use_empty())
      rewriter.eraseOp(softmaxOp);

    return success();
  }

private:
  int64_t tileSize;
};

} // namespace

void mlir::linalg::populateOnlineSoftmaxPatterns(RewritePatternSet &patterns,
                                                  int64_t tileSize) {
  patterns.add<SoftmaxMatmulToOnlineSoftmax>(patterns.getContext(), tileSize);
}
