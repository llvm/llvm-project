//===- SoftmaxMatmulFusion.cpp - Rewrite softmax+matmul to online softmax -------===//
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

#define DEBUG_TYPE "linalg-softmax-matmul-fusion"

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Find a matmul/batch_matmul user of the softmax result where:
/// - The softmax result is the LHS (input 0) of the matmul
/// - The softmax dimension matches the matmul contraction dimension (last dim of LHS)
static Operation *findMatchingMatmulUser(linalg::SoftmaxOp softmaxOp) {
  Value softmaxResult = softmaxOp.getResult()[0];
  int64_t softmaxDim = softmaxOp.getDimension();

  for (Operation *user : softmaxResult.getUsers()) {
    // Match either matmul or batch_matmul.
    auto matmulOp = dyn_cast<linalg::MatmulOp>(user);
    auto batchMatmulOp = dyn_cast<linalg::BatchMatmulOp>(user);
    if (!matmulOp && !batchMatmulOp)
      continue;

    // Get the LHS input.
    Value lhs = matmulOp ? matmulOp.getInputs()[0]
                         : batchMatmulOp.getInputs()[0];
    if (lhs != softmaxResult)
      continue;

    // Contraction dim is the last dim of LHS for both matmul and batch_matmul.
    auto lhsType = cast<RankedTensorType>(softmaxResult.getType());
    int64_t contractionDim = lhsType.getRank() - 1;

    if (softmaxDim == contractionDim)
      return user;
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
struct SoftmaxMatmulToSoftmaxMatmulFusion
    : public OpRewritePattern<linalg::SoftmaxOp> {
  SoftmaxMatmulToSoftmaxMatmulFusion(MLIRContext *ctx, int64_t tileSize)
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

    // Find a matching matmul/batch_matmul user.
    Operation *matmulOp = findMatchingMatmulUser(softmaxOp);
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

    int64_t inputRank = inputType.getRank();

    // Require softmax on last dimension and rank 2 or 3 with all static shapes.
    if (softmaxDim != inputRank - 1)
      return rewriter.notifyMatchFailure(softmaxOp,
                                         "softmax must be on last dim");
    if (inputRank < 2 || inputRank > 3)
      return rewriter.notifyMatchFailure(softmaxOp,
                                         "only rank-2 or rank-3 supported");

    // Collect batch dims + M dim (all dims before the softmax dim).
    // For rank-2: batchAndM = [M]. For rank-3: batchAndM = [B, M].
    SmallVector<int64_t> batchAndM;
    for (int64_t i = 0; i < softmaxDim; ++i) {
      int64_t d = inputType.getShape()[i];
      if (ShapedType::isDynamic(d))
        return rewriter.notifyMatchFailure(softmaxOp, "dynamic batch/M dim");
      batchAndM.push_back(d);
    }
    int64_t M = batchAndM.back();

    // Get V (RHS of the matmul) and Kv (last dim of V).
    Value V = matmulOp->getOperand(1);
    auto vType = cast<RankedTensorType>(V.getType());
    if (vType.getRank() != inputRank)
      return rewriter.notifyMatchFailure(matmulOp, "V rank mismatch");
    int64_t Kv = vType.getShape()[vType.getRank() - 1];
    if (ShapedType::isDynamic(Kv))
      return rewriter.notifyMatchFailure(matmulOp, "Kv dim is dynamic");

    Type elemType = inputType.getElementType();
    Location loc = softmaxOp.getLoc();

    // --- Rewrite ---

    // --- Rewrite ---
    // Shapes: input is [...batch, M, N], softmax on last dim (N).
    // After expand_shape: [...batch, M, tn, ts]
    // m, l shapes: [...batch, M, tn]
    // V shape: [...batch, N, Kv] -> [...batch, tn, ts, Kv]
    // O shape: [...batch, M, Kv]

    MLIRContext *ctx = rewriter.getContext();

    // Compute expanded S shape: [...batch, M, tn, ts]
    SmallVector<int64_t> expandedSShape(batchAndM);
    expandedSShape.push_back(tn);
    expandedSShape.push_back(ts);

    // Compute m/l shape: [...batch, M, tn]
    SmallVector<int64_t> mlShape(batchAndM);
    mlShape.push_back(tn);

    // Compute O shape: [...batch, M, Kv]
    SmallVector<int64_t> oShape(batchAndM);
    oShape.push_back(Kv);

    // Number of dims for the local softmax generics (all batch + M + tn + ts)
    int64_t numLocalDims = expandedSShape.size(); // e.g. 3 for 2D, 4 for 3D

    // Build affine dim exprs for local softmax generics.
    SmallVector<AffineExpr> allDims;
    for (int64_t i = 0; i < numLocalDims; ++i)
      allDims.push_back(rewriter.getAffineDimExpr(i));

    // fullMap: (batch..., M, tn, ts) -> (batch..., M, tn, ts) — identity
    AffineMap fullMap = AffineMap::get(numLocalDims, 0, allDims, ctx);
    // reducedMap: (batch..., M, tn, ts) -> (batch..., M, tn) — drop last
    SmallVector<AffineExpr> reducedExprs(allDims.begin(), allDims.end() - 1);
    AffineMap reducedMap = AffineMap::get(numLocalDims, 0, reducedExprs, ctx);

    // Iterator types: all parallel except last (ts) which varies
    SmallVector<utils::IteratorType> allParallel(numLocalDims,
                                                  utils::IteratorType::parallel);
    SmallVector<utils::IteratorType> lastReduction(allParallel);
    lastReduction.back() = utils::IteratorType::reduction;

    // (a) Reshape S: [..., M, N] -> [..., M, tn, ts] via expand_shape.
    auto expandedSType = RankedTensorType::get(expandedSShape, elemType);
    SmallVector<ReassociationIndices> sReassoc;
    for (int64_t i = 0; i < inputRank - 1; ++i)
      sReassoc.push_back({static_cast<int>(i)});
    sReassoc.push_back(
        {static_cast<int>(inputRank - 1), static_cast<int>(inputRank)});
    Value S_tiled = tensor::ExpandShapeOp::create(rewriter, loc, expandedSType,
                                                  softmaxInput, sReassoc);

    // (b) Compute per-tile max: m[..., M, tn] = max over ts
    Value negInfScalar = arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getFloatAttr(
            elemType, APFloat::getInf(
                          cast<FloatType>(elemType).getFloatSemantics(), true)));
    Value m_init = createFilledTensor(rewriter, loc, mlShape, elemType, negInfScalar);

    auto maxGeneric = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{RankedTensorType::get(mlShape, elemType)},
        /*inputs=*/ValueRange{S_tiled},
        /*outputs=*/ValueRange{m_init},
        SmallVector<AffineMap>{fullMap, reducedMap},
        lastReduction,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = arith::MaxNumFOp::create(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, result);
        });
    Value m = maxGeneric.getResult(0);

    // (c) Compute num = exp(S_tiled - m): elementwise
    Value num_init = tensor::EmptyOp::create(rewriter, loc, expandedSShape, elemType)
                         .getResult();
    auto expGeneric = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{expandedSType},
        /*inputs=*/ValueRange{S_tiled, m},
        /*outputs=*/ValueRange{num_init},
        SmallVector<AffineMap>{fullMap, reducedMap, fullMap},
        allParallel,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value diff = arith::SubFOp::create(b, nestedLoc, args[0], args[1]);
          Value result = math::ExpOp::create(b, nestedLoc, diff);
          linalg::YieldOp::create(b, nestedLoc, result);
        });
    Value num = expGeneric.getResult(0);

    // (d) Compute per-tile sum: l[..., M, tn] = sum over ts
    Value zeroScalar = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));
    Value l_init = createFilledTensor(rewriter, loc, mlShape, elemType, zeroScalar);

    auto sumGeneric = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{RankedTensorType::get(mlShape, elemType)},
        /*inputs=*/ValueRange{num},
        /*outputs=*/ValueRange{l_init},
        SmallVector<AffineMap>{fullMap, reducedMap},
        lastReduction,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = arith::AddFOp::create(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, result);
        });
    Value l = sumGeneric.getResult(0);

    // (e) Compute P = num / l: elementwise
    Value P_init = tensor::EmptyOp::create(rewriter, loc, expandedSShape, elemType)
                       .getResult();
    auto divGeneric = linalg::GenericOp::create(
        rewriter, loc,
        TypeRange{expandedSType},
        /*inputs=*/ValueRange{num, l},
        /*outputs=*/ValueRange{P_init},
        SmallVector<AffineMap>{fullMap, reducedMap, fullMap},
        allParallel,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value result = arith::DivFOp::create(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, result);
        });
    Value P = divGeneric.getResult(0);

    // (f) Reshape V: [...batch, N, Kv] -> [...batch, tn, ts, Kv]
    SmallVector<int64_t> expandedVShape;
    SmallVector<ReassociationIndices> vReassoc;
    // Copy batch dims (if any).
    for (int64_t i = 0; i < inputRank - 2; ++i) {
      expandedVShape.push_back(vType.getShape()[i]);
      vReassoc.push_back({static_cast<int>(i)});
    }
    // Split the N dim into [tn, ts].
    expandedVShape.push_back(tn);
    expandedVShape.push_back(ts);
    vReassoc.push_back({static_cast<int>(inputRank - 2),
                        static_cast<int>(inputRank - 1)});
    // Keep Kv.
    expandedVShape.push_back(Kv);
    vReassoc.push_back({static_cast<int>(inputRank)});

    auto expandedVType = RankedTensorType::get(expandedVShape, elemType);
    Value V_tiled =
        tensor::ExpandShapeOp::create(rewriter, loc, expandedVType, V, vReassoc);

    // (g) Create init tensors for rescaling matmul:
    //     O: [...batch, M, Kv], M_run: [...batch, M, Kv], L_run: [...batch, M, Kv]
    Value O_init =
        createFilledTensor(rewriter, loc, oShape, elemType, zeroScalar);
    Value M_init =
        createFilledTensor(rewriter, loc, oShape, elemType, negInfScalar);
    Value L_init =
        createFilledTensor(rewriter, loc, oShape, elemType, zeroScalar);

    // (h) Build the rescaling matmul linalg.generic.
    // Dimensions: (batch..., m, tn, ts, kv)
    //   batch dims = parallel, m = parallel, tn = reduction, ts = reduction, kv = parallel
    int64_t numRescaleDims = static_cast<int64_t>(batchAndM.size()) + 3; // +tn+ts+kv
    SmallVector<AffineExpr> rescaleDims;
    for (int64_t i = 0; i < numRescaleDims; ++i)
      rescaleDims.push_back(rewriter.getAffineDimExpr(i));

    int64_t nBatchAndM = batchAndM.size(); // number of batch+M dims
    // Indices: batch..., M are [0..nBatchAndM-1], tn=nBatchAndM, ts=nBatchAndM+1, kv=nBatchAndM+2
    int64_t tnIdx = nBatchAndM;
    int64_t tsIdx = nBatchAndM + 1;
    int64_t kvIdx = nBatchAndM + 2;

    // P map: (batch..., m, tn, ts, kv) -> (batch..., m, tn, ts)
    SmallVector<AffineExpr> pExprs(rescaleDims.begin(), rescaleDims.begin() + nBatchAndM);
    pExprs.push_back(rescaleDims[tnIdx]);
    pExprs.push_back(rescaleDims[tsIdx]);
    // m/l map: (batch..., m, tn, ts, kv) -> (batch..., m, tn)
    SmallVector<AffineExpr> mlExprs(rescaleDims.begin(), rescaleDims.begin() + nBatchAndM);
    mlExprs.push_back(rescaleDims[tnIdx]);
    // V map: (batch..., m, tn, ts, kv) -> (batch..., tn, ts, kv)
    SmallVector<AffineExpr> vExprs;
    for (int64_t i = 0; i < nBatchAndM - 1; ++i) // batch dims only (exclude M)
      vExprs.push_back(rescaleDims[i]);
    vExprs.push_back(rescaleDims[tnIdx]);
    vExprs.push_back(rescaleDims[tsIdx]);
    vExprs.push_back(rescaleDims[kvIdx]);
    // O/M_run/L_run map: (batch..., m, tn, ts, kv) -> (batch..., m, kv)
    SmallVector<AffineExpr> oExprs(rescaleDims.begin(), rescaleDims.begin() + nBatchAndM);
    oExprs.push_back(rescaleDims[kvIdx]);

    SmallVector<AffineMap> indexingMaps = {
        AffineMap::get(numRescaleDims, 0, pExprs, ctx),  // P
        AffineMap::get(numRescaleDims, 0, mlExprs, ctx), // m
        AffineMap::get(numRescaleDims, 0, mlExprs, ctx), // l
        AffineMap::get(numRescaleDims, 0, vExprs, ctx),  // V
        AffineMap::get(numRescaleDims, 0, oExprs, ctx),  // O
        AffineMap::get(numRescaleDims, 0, oExprs, ctx),  // M_run
        AffineMap::get(numRescaleDims, 0, oExprs, ctx),  // L_run
    };

    SmallVector<utils::IteratorType> iteratorTypes(numRescaleDims,
                                                    utils::IteratorType::parallel);
    iteratorTypes[tnIdx] = utils::IteratorType::reduction;
    iteratorTypes[tsIdx] = utils::IteratorType::reduction;

    auto oType = RankedTensorType::get(oShape, elemType);
    auto rescalingMatmulOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{oType, oType, oType},
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
      // TODO: Generalize rescaling softmax for arbitrary rank.
      // For now, only emit for rank-2 case.
      if (inputRank == 2) {
        // Build the rescaling softmax generic to recover global softmax.
        SmallVector<int64_t> softmaxOutShape(batchAndM);
        softmaxOutShape.push_back(N);
        auto softmaxOutType = RankedTensorType::get(softmaxOutShape, elemType);

        // Create identity tensor: [tn, ts, N]
        Value I_empty =
            tensor::EmptyOp::create(rewriter, loc,
                                    ArrayRef<int64_t>{tn, ts, N}, elemType)
                .getResult();
        AffineExpr i0, i1, i2;
        bindDims(ctx, i0, i1, i2);
        auto identityGeneric = linalg::GenericOp::create(
            rewriter, loc,
            TypeRange{RankedTensorType::get({tn, ts, N}, elemType)},
            /*inputs=*/ValueRange{},
            /*outputs=*/ValueRange{I_empty},
            SmallVector<AffineMap>{AffineMap::get(3, 0, {i0, i1, i2}, ctx)},
            SmallVector<utils::IteratorType>(3, utils::IteratorType::parallel),
            [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
              Value tIdx = linalg::IndexOp::create(b, nestedLoc, 0);
              Value sIdx = linalg::IndexOp::create(b, nestedLoc, 1);
              Value nIdx = linalg::IndexOp::create(b, nestedLoc, 2);
              Value tsConst = arith::ConstantIndexOp::create(b, nestedLoc, ts);
              Value tTimesTs = arith::MulIOp::create(b, nestedLoc, tIdx, tsConst);
              Value globalIdx = arith::AddIOp::create(b, nestedLoc, tTimesTs, sIdx);
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

        Value Os_init = createFilledTensor(rewriter, loc, softmaxOutShape, elemType, zeroScalar);
        Value Ms_init = createFilledTensor(rewriter, loc, softmaxOutShape, elemType, negInfScalar);
        Value Ls_init = createFilledTensor(rewriter, loc, softmaxOutShape, elemType, zeroScalar);

        // Reuse the same indexing maps/iteratorTypes as the rescaling matmul
        // (they have the same structure: P, m, l, matrix -> O, M_run, L_run)
        auto rescalingSoftmaxOp = linalg::GenericOp::create(
            rewriter, loc,
            TypeRange{softmaxOutType, softmaxOutType, softmaxOutType},
            /*inputs=*/ValueRange{P, m, l, I_tiled},
            /*outputs=*/ValueRange{Os_init, Ms_init, Ls_init}, indexingMaps,
            iteratorTypes, buildRescalingBody);

        Value recoveredSoftmax = rescalingSoftmaxOp.getResult(0);
        rewriter.replaceAllUsesExcept(softmaxResult, recoveredSoftmax, matmulOp);
      }
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

void mlir::linalg::populateSoftmaxMatmulFusionPatterns(RewritePatternSet &patterns,
                                                  int64_t tileSize) {
  patterns.add<SoftmaxMatmulToSoftmaxMatmulFusion>(patterns.getContext(), tileSize);
}
