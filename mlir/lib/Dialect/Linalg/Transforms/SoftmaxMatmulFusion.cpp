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

    // Get the LHS input (first DPS input operand).
    auto dpsOp = cast<linalg::LinalgOp>(user);
    Value lhs = dpsOp.getDpsInputs()[0];
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

/// Build the online-softmax recurrence body for the rescale-and-reduce generic
/// (op2). It reduces over the tile dimension `tn`, combining the per-tile
/// matmul partial `pv = sum_ts(num * V)` and the per-tile denominator
/// `lsum = sum_ts(num)` into the running output O, max M, and denominator L.
///
/// The matmul (sum over ts of num*V) is a SEPARATE preceding generic (op1, a
/// pure contraction that lowers to vector.contract). This body therefore
/// contains no contraction — only the per-tile rescaling. This is the key
/// deviation from Triton's `acc += dot(exp(qk - m_ij), v)`: there the matmul
/// operand is shifted by the running max `m_ij`, which couples it to the
/// recurrence. We instead contract `num = exp(S - m_local)` (local-max shifted,
/// loop-invariant) in op1 and apply the running-max correction factor `beta`
/// here, using the identity sum_ts(num*beta*V) = beta*sum_ts(num*V) (beta is
/// constant over ts). Same result, but op1 is a standalone contraction.
///
/// Per tile (reduce over tn):
///   M_new = max(M_acc, m_local)
///   alpha = exp(M_acc - M_new)         // rescale prior running state
///   beta  = exp(m_local - M_new)       // rebase this tile: local -> running max
///   O_new = O_acc * alpha + beta * pv
///   L_new = L_acc * alpha + beta * lsum
///
/// Args layout: [pv_val, m_tile, lsum_tile, O_acc, M_acc, L_acc]
static void buildRescaleReduceBody(OpBuilder &b, Location loc, ValueRange args) {
  Value pv_val = args[0], m_tile = args[1], lsum_tile = args[2];
  Value O_acc = args[3], M_acc = args[4], L_acc = args[5];

  // M_new = max(M_acc, m_local)
  Value M_new = arith::MaximumFOp::create(b, loc, M_acc, m_tile);

  // Correction factors.
  Value diff1 = arith::SubFOp::create(b, loc, M_acc, M_new);
  Value alpha = math::ExpOp::create(b, loc, diff1); // exp(M_acc - M_new)
  Value diff2 = arith::SubFOp::create(b, loc, m_tile, M_new);
  Value beta = math::ExpOp::create(b, loc, diff2); // exp(m_local - M_new)

  // O_new = O_acc * alpha + beta * pv
  Value O_rescaled = arith::MulFOp::create(b, loc, O_acc, alpha);
  Value O_contrib = arith::MulFOp::create(b, loc, beta, pv_val);
  Value O_new = arith::AddFOp::create(b, loc, O_rescaled, O_contrib);

  // L_new = L_acc * alpha + beta * lsum
  Value L_rescaled = arith::MulFOp::create(b, loc, L_acc, alpha);
  Value L_contrib = arith::MulFOp::create(b, loc, beta, lsum_tile);
  Value L_new = arith::AddFOp::create(b, loc, L_rescaled, L_contrib);

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

    // No per-tile normalization: the rescaling matmul consumes the unnormalized
    // num = exp(S - m_local) directly (matching Triton, which feeds p into the
    // dot). `l` is still produced above — it is only needed by the global
    // softmax recovery path below.

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

    // (h) Build the online-softmax matmul as THREE ops so the matmul is a
    //     standalone contraction (-> vector.contract):
    //   op1  (pv):   pv[batch.., m, tn, kv] = sum_ts num * V    (ts reduction,
    //                tn parallel) — a pure multiply-accumulate contraction.
    //   op1b (lsum): lsum[batch.., m, tn]   = sum_ts num        (ts reduction)
    //   op2:         online recurrence reduced over tn (running max/rescale),
    //                consuming pv, m, lsum (see buildRescaleReduceBody).
    //   op3:         final divide O = O / L.
    int64_t nBatchAndM = batchAndM.size(); // number of batch+M dims
    auto oType = RankedTensorType::get(oShape, elemType);

    // partial (pv) shape: [batch..., M, tn, Kv]
    SmallVector<int64_t> pvShape(batchAndM);
    pvShape.push_back(tn);
    pvShape.push_back(Kv);
    auto pvType = RankedTensorType::get(pvShape, elemType);

    // --- op1 (pv): contraction over ts, tn parallel ---
    // Dimensions: (batch..., m, tn, ts, kv)
    int64_t numMmDims = nBatchAndM + 3; // +tn+ts+kv
    SmallVector<AffineExpr> mmDims;
    for (int64_t i = 0; i < numMmDims; ++i)
      mmDims.push_back(rewriter.getAffineDimExpr(i));
    int64_t mm_tn = nBatchAndM, mm_ts = nBatchAndM + 1, mm_kv = nBatchAndM + 2;
    // num: (batch..., m, tn, ts)
    SmallVector<AffineExpr> mmNum(mmDims.begin(), mmDims.begin() + nBatchAndM);
    mmNum.push_back(mmDims[mm_tn]);
    mmNum.push_back(mmDims[mm_ts]);
    // V: (batch..., tn, ts, kv)
    SmallVector<AffineExpr> mmV;
    for (int64_t i = 0; i < nBatchAndM - 1; ++i)
      mmV.push_back(mmDims[i]);
    mmV.push_back(mmDims[mm_tn]);
    mmV.push_back(mmDims[mm_ts]);
    mmV.push_back(mmDims[mm_kv]);
    // pv: (batch..., m, tn, kv)
    SmallVector<AffineExpr> mmPv(mmDims.begin(), mmDims.begin() + nBatchAndM);
    mmPv.push_back(mmDims[mm_tn]);
    mmPv.push_back(mmDims[mm_kv]);
    SmallVector<utils::IteratorType> mmIters(numMmDims,
                                             utils::IteratorType::parallel);
    mmIters[mm_ts] = utils::IteratorType::reduction;
    Value pvInit = createFilledTensor(rewriter, loc, pvShape, elemType, zeroScalar);
    auto pvOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{pvType}, ValueRange{num, V_tiled},
        ValueRange{pvInit},
        SmallVector<AffineMap>{AffineMap::get(numMmDims, 0, mmNum, ctx),
                               AffineMap::get(numMmDims, 0, mmV, ctx),
                               AffineMap::get(numMmDims, 0, mmPv, ctx)},
        mmIters, [&](OpBuilder &b, Location l, ValueRange a) {
          Value prod = arith::MulFOp::create(b, l, a[0], a[1]);
          Value sum = arith::AddFOp::create(b, l, a[2], prod);
          linalg::YieldOp::create(b, l, sum);
        });
    Value pv = pvOp.getResult(0);

    // --- op1b (lsum): lsum[batch..., m, tn] = sum_ts num ---
    SmallVector<int64_t> lsumShape(batchAndM);
    lsumShape.push_back(tn);
    auto lsumType = RankedTensorType::get(lsumShape, elemType);
    int64_t numLsumDims = nBatchAndM + 2; // +tn+ts
    SmallVector<AffineExpr> lsDims;
    for (int64_t i = 0; i < numLsumDims; ++i)
      lsDims.push_back(rewriter.getAffineDimExpr(i));
    int64_t ls_tn = nBatchAndM, ls_ts = nBatchAndM + 1;
    SmallVector<AffineExpr> lsNum(lsDims.begin(), lsDims.begin() + nBatchAndM);
    lsNum.push_back(lsDims[ls_tn]);
    lsNum.push_back(lsDims[ls_ts]);
    SmallVector<AffineExpr> lsOut(lsDims.begin(), lsDims.begin() + nBatchAndM);
    lsOut.push_back(lsDims[ls_tn]);
    SmallVector<utils::IteratorType> lsIters(numLsumDims,
                                             utils::IteratorType::parallel);
    lsIters[ls_ts] = utils::IteratorType::reduction;
    Value lsumInit =
        createFilledTensor(rewriter, loc, lsumShape, elemType, zeroScalar);
    auto lsumOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{lsumType}, ValueRange{num},
        ValueRange{lsumInit},
        SmallVector<AffineMap>{AffineMap::get(numLsumDims, 0, lsNum, ctx),
                               AffineMap::get(numLsumDims, 0, lsOut, ctx)},
        lsIters, [&](OpBuilder &b, Location l, ValueRange a) {
          Value sum = arith::AddFOp::create(b, l, a[1], a[0]);
          linalg::YieldOp::create(b, l, sum);
        });
    Value lsum = lsumOp.getResult(0);

    // --- op2: online recurrence reduced over tn ---
    // Dimensions: (batch..., m, tn, kv); reduce tn.
    int64_t numRecDims = nBatchAndM + 2; // +tn+kv
    SmallVector<AffineExpr> recDims;
    for (int64_t i = 0; i < numRecDims; ++i)
      recDims.push_back(rewriter.getAffineDimExpr(i));
    int64_t rec_tn = nBatchAndM, rec_kv = nBatchAndM + 1;
    // pv: (batch..., m, tn, kv)
    SmallVector<AffineExpr> recPv(recDims.begin(), recDims.begin() + nBatchAndM);
    recPv.push_back(recDims[rec_tn]);
    recPv.push_back(recDims[rec_kv]);
    // m/lsum: (batch..., m, tn)
    SmallVector<AffineExpr> recMl(recDims.begin(), recDims.begin() + nBatchAndM);
    recMl.push_back(recDims[rec_tn]);
    // O/M/L: (batch..., m, kv)
    SmallVector<AffineExpr> recO(recDims.begin(), recDims.begin() + nBatchAndM);
    recO.push_back(recDims[rec_kv]);
    SmallVector<utils::IteratorType> recIters(numRecDims,
                                              utils::IteratorType::parallel);
    recIters[rec_tn] = utils::IteratorType::reduction;
    auto recOp = linalg::GenericOp::create(
        rewriter, loc,
        /*resultTypes=*/TypeRange{oType, oType, oType},
        /*inputs=*/ValueRange{pv, m, lsum},
        /*outputs=*/ValueRange{O_init, M_init, L_init},
        SmallVector<AffineMap>{AffineMap::get(numRecDims, 0, recPv, ctx),
                               AffineMap::get(numRecDims, 0, recMl, ctx),
                               AffineMap::get(numRecDims, 0, recMl, ctx),
                               AffineMap::get(numRecDims, 0, recO, ctx),
                               AffineMap::get(numRecDims, 0, recO, ctx),
                               AffineMap::get(numRecDims, 0, recO, ctx)},
        recIters, buildRescaleReduceBody);

    // The recurrence keeps O unnormalized; divide by the final denominator L
    // once, here (elementwise over [batch..., M, Kv]).
    Value A_final = recOp.getResult(0); // unnormalized output
    Value L_final = recOp.getResult(2); // denominator
    int64_t numDivDims = oShape.size();
    SmallVector<AffineExpr> divDims;
    for (int64_t i = 0; i < numDivDims; ++i)
      divDims.push_back(rewriter.getAffineDimExpr(i));
    AffineMap divMap = AffineMap::get(numDivDims, 0, divDims, ctx);
    SmallVector<utils::IteratorType> divIters(numDivDims,
                                              utils::IteratorType::parallel);
    Value O_div_init =
        tensor::EmptyOp::create(rewriter, loc, oShape, elemType).getResult();
    auto divideOp = linalg::GenericOp::create(
        rewriter, loc, TypeRange{oType},
        /*inputs=*/ValueRange{A_final, L_final},
        /*outputs=*/ValueRange{O_div_init},
        SmallVector<AffineMap>{divMap, divMap, divMap}, divIters,
        [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value q = arith::DivFOp::create(b, nestedLoc, args[0], args[1]);
          linalg::YieldOp::create(b, nestedLoc, q);
        });

    Value rescaledO = divideOp.getResult(0);

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
      // Recover global softmax from (P, m, l) using two generics + collapse_shape:
      //
      // Generic 1: Reduce (m, l) over tn to get M_global[..., M] and L_global[..., M]
      //   M_global = max over all tn of m[..., tn]
      //   L_global = sum over all tn of l[..., tn] * exp(m[..., tn] - M_global)
      //
      // Generic 2: Elementwise correction of P
      //   corrected_P[..., tn, ts] = P[..., tn, ts] * l[..., tn] * exp(m[..., tn] - M_global) / L_global
      //
      // collapse_shape: [..., tn, ts] -> [..., N]

      // Shapes: mGlobalShape = [...batch, M], same as batchAndM
      SmallVector<int64_t> mGlobalShape(batchAndM);
      auto mGlobalType = RankedTensorType::get(mGlobalShape, elemType);

      // --- Generic 1: Compute M_global and L_global via reduction over tn ---
      // Dims: (batch..., m, tn) with tn as reduction
      int64_t numReduceDims = mlShape.size(); // [...batch, M, tn]
      SmallVector<AffineExpr> reduceDims;
      for (int64_t i = 0; i < numReduceDims; ++i)
        reduceDims.push_back(rewriter.getAffineDimExpr(i));

      // Input map (m, l): identity over all dims [..., M, tn]
      AffineMap reduceFullMap = AffineMap::get(numReduceDims, 0, reduceDims, ctx);
      // Output map (M_global, L_global): drop last dim (tn)
      SmallVector<AffineExpr> reduceOutExprs(reduceDims.begin(), reduceDims.end() - 1);
      AffineMap reduceOutMap = AffineMap::get(numReduceDims, 0, reduceOutExprs, ctx);

      SmallVector<utils::IteratorType> reduceIters(numReduceDims,
                                                    utils::IteratorType::parallel);
      reduceIters.back() = utils::IteratorType::reduction;

      Value Mg_init = createFilledTensor(rewriter, loc, mGlobalShape, elemType, negInfScalar);
      Value Lg_init = createFilledTensor(rewriter, loc, mGlobalShape, elemType, zeroScalar);

      auto globalReduceOp = linalg::GenericOp::create(
          rewriter, loc,
          TypeRange{mGlobalType, mGlobalType},
          /*inputs=*/ValueRange{m, l},
          /*outputs=*/ValueRange{Mg_init, Lg_init},
          SmallVector<AffineMap>{reduceFullMap, reduceFullMap, reduceOutMap, reduceOutMap},
          reduceIters,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value m_i = args[0], l_i = args[1], Mg_acc = args[2], Lg_acc = args[3];
            // M_new = max(Mg_acc, m_i)
            Value Mg_new = arith::MaxNumFOp::create(b, nestedLoc, Mg_acc, m_i);
            // L_new = Lg_acc * exp(Mg_acc - Mg_new) + l_i * exp(m_i - Mg_new)
            Value diff1 = arith::SubFOp::create(b, nestedLoc, Mg_acc, Mg_new);
            Value corr = math::ExpOp::create(b, nestedLoc, diff1);
            Value Lg_rescaled = arith::MulFOp::create(b, nestedLoc, Lg_acc, corr);
            Value diff2 = arith::SubFOp::create(b, nestedLoc, m_i, Mg_new);
            Value exp2 = math::ExpOp::create(b, nestedLoc, diff2);
            Value contrib = arith::MulFOp::create(b, nestedLoc, l_i, exp2);
            Value Lg_new = arith::AddFOp::create(b, nestedLoc, Lg_rescaled, contrib);
            linalg::YieldOp::create(b, nestedLoc, ValueRange{Mg_new, Lg_new});
          });
      Value M_global = globalReduceOp.getResult(0);
      Value L_global = globalReduceOp.getResult(1);

      // --- Generic 2: Correct P elementwise ---
      // corrected_P[..., M, tn, ts] = P[..., M, tn, ts] * l[..., M, tn] * exp(m[..., M, tn] - M_global[..., M]) / L_global[..., M]
      // Dims: (batch..., M, tn, ts) — all parallel
      // expandedSShape = [...batch, M, tn, ts]
      auto correctedType = RankedTensorType::get(expandedSShape, elemType);
      Value corrected_init = tensor::EmptyOp::create(rewriter, loc, expandedSShape, elemType).getResult();

      // Maps for the correction generic:
      // P:        fullMap = identity over all dims
      // l:        reducedMap = [..., M, tn] (drop ts)
      // m:        reducedMap = [..., M, tn] (drop ts)
      // M_global: [..., M] (drop tn and ts)
      // L_global: [..., M] (drop tn and ts)
      // output:   fullMap = identity
      SmallVector<AffineExpr> globalExprs(allDims.begin(), allDims.end() - 2); // drop tn and ts
      AffineMap globalMap = AffineMap::get(numLocalDims, 0, globalExprs, ctx);

      // softmax[..., tn, ts] = num * exp(m - Mg) / Lg.
      // (Since num = exp(S - m_local) and P = num/l_local, the local l cancels:
      //  P * l * exp(m - Mg)/Lg = num * exp(m - Mg)/Lg.)
      auto correctionOp = linalg::GenericOp::create(
          rewriter, loc,
          TypeRange{correctedType},
          /*inputs=*/ValueRange{num, m, M_global, L_global},
          /*outputs=*/ValueRange{corrected_init},
          SmallVector<AffineMap>{fullMap, reducedMap, globalMap, globalMap, fullMap},
          allParallel,
          [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
            Value num_v = args[0], m_i = args[1];
            Value Mg = args[2], Lg = args[3];
            // w = exp(m_i - Mg) / Lg
            Value diff = arith::SubFOp::create(b, nestedLoc, m_i, Mg);
            Value expDiff = math::ExpOp::create(b, nestedLoc, diff);
            Value w = arith::DivFOp::create(b, nestedLoc, expDiff, Lg);
            // corrected = num * w
            Value result = arith::MulFOp::create(b, nestedLoc, num_v, w);
            linalg::YieldOp::create(b, nestedLoc, result);
          });
      Value correctedP = correctionOp.getResult(0);

      // --- collapse_shape: [..., M, tn, ts] -> [..., M, N] ---
      SmallVector<int64_t> softmaxOutShape(batchAndM);
      softmaxOutShape.push_back(N);
      auto softmaxOutType = RankedTensorType::get(softmaxOutShape, elemType);
      // Reassociation: keep batch+M dims as-is, merge [tn, ts] into one dim.
      SmallVector<ReassociationIndices> collapseReassoc;
      for (int64_t i = 0; i < static_cast<int64_t>(batchAndM.size()); ++i)
        collapseReassoc.push_back({static_cast<int>(i)});
      collapseReassoc.push_back({static_cast<int>(batchAndM.size()),
                                 static_cast<int>(batchAndM.size() + 1)});

      Value recoveredSoftmax = tensor::CollapseShapeOp::create(
          rewriter, loc, softmaxOutType, correctedP, collapseReassoc);

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

void mlir::linalg::populateSoftmaxMatmulFusionPatterns(RewritePatternSet &patterns,
                                                  int64_t tileSize) {
  patterns.add<SoftmaxMatmulToSoftmaxMatmulFusion>(patterns.getContext(), tileSize);
}
