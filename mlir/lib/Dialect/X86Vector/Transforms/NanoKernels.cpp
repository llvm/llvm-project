//===- NanoKernels.cpp - Lower matmul to Nanokernels -- -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements matmul rewrites as nanokernels with respect to target
// machine for FP32 and BF16 (TODO) types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86vector;

static FailureOr<SmallVector<scf::ForOp>>
getNestedLoop(vector::ContractionOp contractOp, unsigned int dimCount) {
  SmallVector<scf::ForOp> list;
  Operation *current = contractOp;
  // It is register tiled loop structure on batch reduce matmul
  // (M->N->Batch-reduce->K).
  for (unsigned int i = 0; i < dimCount; i++) {
    Operation *parent = current->getParentOfType<scf::ForOp>();
    if (!parent)
      return failure();
    list.push_back(dyn_cast<scf::ForOp>(parent));
    current = parent;
  }
  return list;
}

static LogicalResult checkNestedLoop(SmallVector<scf::ForOp> loops,
                                     SmallVector<memref::SubViewOp> subviews,
                                     unsigned int dimCount) {
  auto subviewOpLhsOffsets = subviews[0].getOffsets();
  auto subviewOpRhsOffsets = subviews[1].getOffsets();
  auto subviewOpAccOffsets = subviews[2].getOffsets();

  if (dimCount == 4) {
    Value ivK = loops[0].getInductionVar();
    if (ivK != subviewOpLhsOffsets[2] || ivK != subviewOpRhsOffsets[1])
      return failure();

    Value ivReduction = loops[1].getInductionVar();
    if (ivReduction != subviewOpLhsOffsets[0] ||
        ivReduction != subviewOpRhsOffsets[0])
      return failure();

    Value ivN = loops[2].getInductionVar();
    if (ivN != subviewOpAccOffsets[1] || ivN != subviewOpRhsOffsets[2])
      return failure();

    Value ivM = loops[3].getInductionVar();
    if (ivM != subviewOpLhsOffsets[1] || ivM != subviewOpAccOffsets[0])
      return failure();
  }

  if (dimCount == 3) {
    Value ivK = loops[0].getInductionVar();
    if (ivK != subviewOpLhsOffsets[1] || ivK != subviewOpRhsOffsets[0])
      return failure();

    Value ivN = loops[1].getInductionVar();
    if (ivN != subviewOpAccOffsets[1] || ivN != subviewOpRhsOffsets[1])
      return failure();

    Value ivM = loops[2].getInductionVar();
    if (ivM != subviewOpLhsOffsets[0] || ivM != subviewOpAccOffsets[0])
      return failure();
  }

  return success();
}

static SmallVector<Value> loadAcc(Location loc, RewriterBase &rewriter,
                                  Type elementType, unsigned int M,
                                  unsigned int N, unsigned int vectorSize,
                                  Value subviewOpAcc) {

  SmallVector<Value> loopItrArgs;
  unsigned int outerBound = M;
  unsigned int innerBound = N;

  unsigned int outerStep = 1;
  unsigned int innerStep = vectorSize;

  if ((N / vectorSize) > M) {
    outerBound = N;
    innerBound = M;

    outerStep = vectorSize;
    innerStep = 1;
  }

  for (unsigned int i = 0; i < outerBound; i = i + outerStep) {
    for (unsigned int j = 0; j < innerBound; j = j + innerStep) {
      Value indexOp_A = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value indexOp_B = rewriter.create<arith::ConstantIndexOp>(loc, j);

      if ((N / vectorSize) > M) {
        indexOp_A = indexOp_B;
        indexOp_B = rewriter.create<arith::ConstantIndexOp>(loc, i);
      }

      auto valueCRow = rewriter.create<vector::LoadOp>(
          loc, VectorType::get(vectorSize, elementType), subviewOpAcc,
          ValueRange{indexOp_A, indexOp_B});
      loopItrArgs.push_back(valueCRow);
    }
  }

  return loopItrArgs;
}

SmallVector<Value> nanoKernels(RewriterBase &rewriter, Location loc,
                               Type elementType, unsigned int vectorSize,
                               unsigned int vnni, unsigned int M,
                               unsigned int N, ValueRange acc, Value matA,
                               Value matB, unsigned int dimCount) {

  SmallVector<Value> accVector;
  SmallVector<Value> matLoad;
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  unsigned int outerBound = M;
  unsigned int outerStep = 1;

  unsigned int innerBound = N;
  unsigned int innerStep = vectorSize;

  Value outerMatrix = matA;
  Value innerMatrix = matB;

  unsigned int outerVectSize = vnni;
  unsigned int innerVectSize = vectorSize;

  unsigned int fmaBound = M;

  if ((N / vectorSize) < M) {
    outerBound = N;
    innerBound = M;

    outerStep = vectorSize;
    innerStep = 1;

    outerMatrix = matB;
    innerMatrix = matA;

    outerVectSize = vectorSize;
    innerVectSize = vnni;

    fmaBound = N / vectorSize;
  }

  for (unsigned int i = 0; i < outerBound; i = i + outerStep) {
    Value indexOp_i = rewriter.create<arith::ConstantIndexOp>(loc, i);
    Value valueRow;

    if ((N / vectorSize) > M) {

      SmallVector<Value> index = {c0, indexOp_i, c0};
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      Value row = rewriter.create<vector::LoadOp>(
          loc, VectorType::get(outerVectSize, elementType), outerMatrix, index);
      valueRow = rewriter.create<vector::BroadcastOp>(
          loc, VectorType::get(vectorSize, rewriter.getF32Type()), row);
    } else {

      SmallVector<Value> index = {c0, c0, indexOp_i};
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      valueRow = rewriter.create<vector::LoadOp>(
          loc, VectorType::get(outerVectSize, elementType), outerMatrix, index);
    }

    matLoad.push_back(valueRow);
  }

  for (unsigned int j = 0, k = 0; j < innerBound; j = j + innerStep) {
    Value indexOp_j = rewriter.create<arith::ConstantIndexOp>(loc, j);
    Value valueRow;

    if ((N / vectorSize) < M) {
      SmallVector<Value> index = {c0, indexOp_j, c0};
      if (dimCount == 3) {
        index.erase(index.begin());
      }
      Value row = rewriter.create<vector::LoadOp>(
          loc, VectorType::get(innerVectSize, elementType), innerMatrix,
          ValueRange(index));
      valueRow = rewriter.create<vector::BroadcastOp>(
          loc, VectorType::get(vectorSize, rewriter.getF32Type()), row);
    } else {

      SmallVector<Value> index = {c0, c0, indexOp_j};
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      valueRow = rewriter.create<vector::LoadOp>(
          loc, VectorType::get(innerVectSize, elementType), innerMatrix, index);
    }

    for (unsigned int i = 0; i < fmaBound; i = i + 1) {
      auto fmaOdd =
          rewriter.create<vector::FMAOp>(loc, matLoad[i], valueRow, acc[k]);
      k++;
      accVector.push_back(fmaOdd);
    }
  }

  return accVector;
}

Value accVector(RewriterBase &rewriter, Location loc, VectorType vecType,
                SmallVector<Value> FMAs, Value accVec, unsigned int vecSize,
                unsigned int M, unsigned int N) {

  auto strides = rewriter.getI64ArrayAttr({1});
  if ((N / vecSize) > M) {
    for (unsigned int j = 0, k = 0; j < (N / vecSize); j++) {
      for (unsigned int i = 0; i < M; i++) {
        unsigned int off = (j * vecSize) + (i * N);
        auto offsets = rewriter.getI64ArrayAttr({off});
        accVec = rewriter.create<vector::InsertStridedSliceOp>(
            loc, vecType, FMAs[k], accVec, offsets, strides);
        k++;
      }
    }

  } else {
    for (unsigned int i = 0, k = 0; i < M * N; i = i + vecSize) {
      auto offsets = rewriter.getI64ArrayAttr({i});
      accVec = rewriter.create<vector::InsertStridedSliceOp>(
          loc, vecType, FMAs[k], accVec, offsets, strides);
      k++;
    }
  }
  return accVec;
}

scf::ForOp createLoop(RewriterBase &rewriter, Location loc, scf::ForOp kForOp,
                      vector::TransferReadOp vectorReadOpLhs,
                      vector::TransferReadOp vectorReadOpRhs,
                      Value ivNewReductionForOp, Type elementType,
                      unsigned int vectorSize, unsigned int vnni,
                      unsigned int M, unsigned int N,
                      ValueRange iterArgsNewReductionForOp,
                      unsigned int dimCount) {
  auto newKForOp = rewriter.create<scf::ForOp>(
      kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
      kForOp.getStep(), iterArgsNewReductionForOp,
      [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
          Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
        IRMapping mapping;
        mapping.map(vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
        mapping.map(vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
        auto lhsClone = rewriterNewKForOp.clone(
            *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

        IRMapping rhsMapping;
        rhsMapping.map(vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                       ivNewReductionForOp);
        rhsMapping.map(vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                       ivNewKForOp);
        auto rhsClone = rewriterNewKForOp.clone(
            *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

        auto evenFMAs =
            nanoKernels(rewriter, kForOp.getLoc(), elementType, vectorSize,
                        vnni, M, N, iterArgsNewKForOp, lhsClone->getResult(0),
                        rhsClone->getResult(0), dimCount);

        rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, evenFMAs);
      });

  return newKForOp;
}

scf::ForOp createLoop(RewriterBase &rewriter, Location loc, scf::ForOp kForOp,
                      vector::TransferReadOp vectorReadOpLhs,
                      vector::TransferReadOp vectorReadOpRhs, Type elementType,
                      unsigned int vectorSize, unsigned int vnni,
                      unsigned int M, unsigned int N,
                      ValueRange iterArgsNewReductionForOp,
                      unsigned int dimCount) {

  auto newKForOp = rewriter.create<scf::ForOp>(
      kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
      kForOp.getStep(), iterArgsNewReductionForOp,
      [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
          Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
        IRMapping mapping;
        mapping.map(vectorReadOpLhs.getBase().getDefiningOp()->getOperand(2),
                    ivNewKForOp);
        auto lhsClone = rewriterNewKForOp.clone(
            *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

        IRMapping rhsMapping;
        rhsMapping.map(vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                       ivNewKForOp);
        auto rhsClone = rewriterNewKForOp.clone(
            *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

        auto evenFMAs =
            nanoKernels(rewriter, loc, elementType, vectorSize, vnni, M, N,
                        iterArgsNewKForOp, lhsClone->getResult(0),
                        rhsClone->getResult(0), dimCount);

        rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, evenFMAs);
      });

  return newKForOp;
}

struct VectorContractNanokernelLowering
    : public OpRewritePattern<vector::ContractionOp> {
  VectorContractNanokernelLowering(MLIRContext *context,
                                   std::optional<unsigned> vecSize)
      : OpRewritePattern<vector::ContractionOp>(context),
        userVectorSize(vecSize) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    auto loc = contractOp.getLoc();

    unsigned int vectorSize = 8;

    if (userVectorSize)
      vectorSize = *userVectorSize;

    if (contractOp.getKind() != vector::CombiningKind::ADD) {
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");
    }

    auto dimCount = contractOp.getRhsType().getRank() + 1;

    if ((dimCount != 3) && (dimCount != 4))
      return rewriter.notifyMatchFailure(
          contractOp, "Expects batch-reduce or batch matmuls");

    // Get the M, N, K, and batch-reduce loops
    auto loops = getNestedLoop(contractOp, dimCount);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto nestedLoops = *loops;
    scf::ForOp kForOp = nestedLoops[0];
    scf::ForOp reductionForOp;

    vector::TransferReadOp vectorReadOpAcc;

    if (dimCount == 4) {
      reductionForOp = nestedLoops[1];
      vectorReadOpAcc = reductionForOp.getInitArgs()[0]
                            .getDefiningOp<vector::TransferReadOp>();
    }

    if (dimCount == 3) {
      vectorReadOpAcc =
          kForOp.getInitArgs()[0].getDefiningOp<vector::TransferReadOp>();
    }

    auto vectorReadOpLhs =
        contractOp.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto vectorReadOpRhs =
        contractOp.getRhs().getDefiningOp<vector::TransferReadOp>();

    if (!vectorReadOpAcc || !vectorReadOpLhs || !vectorReadOpRhs)
      return failure();

    auto subviewOpAcc =
        vectorReadOpAcc.getOperand(0).getDefiningOp<memref::SubViewOp>();
    auto subviewOpLhs =
        vectorReadOpLhs.getOperand(0).getDefiningOp<memref::SubViewOp>();
    auto subviewOpRhs =
        vectorReadOpRhs.getOperand(0).getDefiningOp<memref::SubViewOp>();

    if (!subviewOpAcc || !subviewOpLhs || !subviewOpRhs)
      return failure();

    SmallVector<memref::SubViewOp> subviews;
    subviews.push_back(subviewOpLhs);
    subviews.push_back(subviewOpRhs);
    subviews.push_back(subviewOpAcc);

    // The M, N, K, and batch-reduce loop iv should match the iv's
    // used in the subviews
    auto checkLoops = checkNestedLoop(*loops, subviews, dimCount);
    if (failed(checkLoops))
      return rewriter.notifyMatchFailure(
          contractOp, "Loops doesn't match the iv in subviews");

    auto elementType =
        (cast<MemRefType>(subviewOpLhs.getType())).getElementType();

    // TODO: Support for BF16 Type
    if (!elementType.isF32())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only, FP32 type is supported");

    auto lhsType = dyn_cast<ShapedType>(vectorReadOpLhs.getType());
    auto rhsType = dyn_cast<ShapedType>(vectorReadOpRhs.getType());

    // Get M, N, and K dimension size
    unsigned int M = lhsType.getDimSize(lhsType.getRank() - 2);
    unsigned int N = rhsType.getDimSize(rhsType.getRank() - 1);
    unsigned int K = lhsType.getDimSize(lhsType.getRank() - 1);
    unsigned int vnni = 1;

    if (K != 1)
      return rewriter.notifyMatchFailure(contractOp, "The k-dim should be 1");

    if (dimCount == 4 && lhsType.getDimSize(lhsType.getRank() - 3) != 1)
      return rewriter.notifyMatchFailure(contractOp,
                                         "The reduction-dim should be 1");

    if (dimCount == 4)
      rewriter.setInsertionPoint(reductionForOp);

    if (dimCount == 3)
      rewriter.setInsertionPoint(kForOp);

    // Load  MxN C sub matrix into acc vectors (e.g, <vectorSizexf32>)
    SmallVector<Value> loopItrArgs =
        loadAcc(loc, rewriter, elementType, M, N, vectorSize, subviewOpAcc);

    // Create the batch-reduce and K-loop with acc vectors as the loop
    // iterargs (batch-reduce matmul) + nanokernel generation
    scf::ForOp newLoop;
    if (dimCount == 4) {
      newLoop = rewriter.create<scf::ForOp>(
          reductionForOp.getLoc(), reductionForOp.getLowerBound(),
          reductionForOp.getUpperBound(), reductionForOp.getStep(), loopItrArgs,
          [&](OpBuilder &rewriterNewReductionForOp,
              Location locNewReductionForOp, Value ivNewReductionForOp,
              ValueRange iterArgsNewReductionForOp) {
            scf::ForOp newKForOp = createLoop(
                rewriter, loc, kForOp, vectorReadOpLhs, vectorReadOpRhs,
                ivNewReductionForOp, elementType, vectorSize, vnni, M, N,
                iterArgsNewReductionForOp, dimCount);

            rewriterNewReductionForOp.create<scf::YieldOp>(
                locNewReductionForOp, newKForOp.getResults());
          });
    }

    // Create only the K-loop (batch matmul) + nanokernel generation
    if (dimCount == 3) {
      newLoop = createLoop(rewriter, loc, kForOp, vectorReadOpLhs,
                           vectorReadOpRhs, elementType, vectorSize, vnni, M, N,
                           loopItrArgs, dimCount);
    }

    // Combine all acc vectors into a MxN C matrix
    auto vecType = VectorType::get({M * N}, rewriter.getF32Type());
    auto zeroAttr =
        DenseElementsAttr::get(vecType, rewriter.getF32FloatAttr(0.0));
    Value accVec = rewriter.create<arith::ConstantOp>(loc, vecType, zeroAttr);

    accVec = accVector(rewriter, loc, vecType, newLoop.getResults(), accVec,
                       vectorSize, M, N);

    auto accTy = dyn_cast<VectorType>(contractOp.getAccType());
    auto reshapeAcc = rewriter.create<vector::ShapeCastOp>(loc, accTy, accVec);

    // Replace all the use of vector.contract with results of nanokernels
    if (dimCount == 4)
      rewriter.replaceAllUsesWith(reductionForOp.getResult(0), reshapeAcc);

    if (dimCount == 3)
      rewriter.replaceAllUsesWith(kForOp.getResult(0), reshapeAcc);

    return success();
  }
  std::optional<unsigned> userVectorSize;
};

void x86vector::populateVectorContractNanokernelLoweringPatterns(
    RewritePatternSet &patterns, std::optional<unsigned> userVectorSize) {
  patterns.add<VectorContractNanokernelLowering>(patterns.getContext(),
                                                 userVectorSize);
}
