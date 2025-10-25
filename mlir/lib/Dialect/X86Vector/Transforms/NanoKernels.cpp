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

static SmallVector<Value>
loadAccumulatorBeforeGEMM(Location loc, RewriterBase &rewriter,
                          Type elementType, unsigned int M, unsigned int N,
                          unsigned int vectorSize, Value subviewOpAcc) {

  SmallVector<Value> accumulators;
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
      Value indexOp_A = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value indexOp_B = arith::ConstantIndexOp::create(rewriter, loc, j);

      if ((N / vectorSize) > M) {
        indexOp_A = indexOp_B;
        indexOp_B = arith::ConstantIndexOp::create(rewriter, loc, i);
      }

      auto valueCRow = vector::LoadOp::create(
          rewriter, loc, VectorType::get(vectorSize, elementType), subviewOpAcc,
          ValueRange{indexOp_A, indexOp_B});
      accumulators.push_back(valueCRow);
    }
  }

  return accumulators;
}

// Function accepts A Matrix, B Matrix, C Matrix (as vectors) and generate
// equivalent target specific nanokernels. Returns the final accumulator as
// output. Based on M tile, N tile, and vector size it generated optimized
// nanokernels with condition of reduction and K dimension of the input matrix
// are 1.
//
// Input: Matrix A, Matrix B, Accmulator as M*(N/vector size) vectors, M tile
// size, N tile size, Vector size.
//
// Output:
// case i: M > (N/vector size). For example, M=3; N=32; vector size = 16.
//  load_B0 = load B[0-15] into vector<16xf32>
//  load_B1 = load B[16-31] into vector<16xf32>
//  bcst_A0 = load A[0] and broadcast it into vector<16xf32>
//  o/p_Acc[0] = vector.fma bcst_A0, load_B0, i/p_Acc[0]
//  o/p_Acc[1] = vector.fma bcst_A0, load_B1, i/p_Acc[1]
//  bcst_A1 = load A[1] and broadcast it into vector<16xf32>
//  o/p_Acc[2] = vector.fma bcst_A1, load_B0, i/p_Acc[2]
//  o/p_Acc[3] = vector.fma bcst_A1, load_B1, i/p_Acc[3]
//  bcst_A2 = load A[2] and broadcast it into vector<16xf32>
//  o/p_Acc[4] = vector.fma bcst_A2, load_B0, i/p_Acc[4]
//  o/p_Acc[5] = vector.fma bcst_A2, load_B1, i/p_Acc[5]
//
// case ii: M <= (N/vector size). For example, M=2; N=48; vector size = 16.
//  bcst_A0 = load A[0] and broadcast it into vector<16xf32>
//  bcst_A1 = load A[1] and broadcast it into vector<16xf32>
//  bcst_A2 = load A[2] and broadcast it into vector<16xf32>
//  load_B0 = load B[0-15] into vector<16xf32>
//  o/p_Acc[0] = vector.fma bcst_A0, load_B0, i/p_Acc[0]
//  o/p_Acc[1] = vector.fma bcst_A1, load_B0, i/p_Acc[1]
//  load_B1 = load B[16-31] into vector<16xf32>
//  o/p_Acc[2] = vector.fma bcst_A0, load_B1, i/p_Acc[2]
//  o/p_Acc[3] = vector.fma bcst_A1, load_B1, i/p_Acc[3]
//  load_B2 = load B[32-47] into vector<16xf32>
//  o/p_Acc[4] = vector.fma bcst_A0, load_B2, i/p_Acc[4]
//  o/p_Acc[5] = vector.fma bcst_A1, load_B2, i/p_Acc[5]
//
// return o/p_Acc;
SmallVector<Value>
generateNanokernels(RewriterBase &rewriter, Location loc, Type elementType,
                    unsigned int vectorSize, unsigned int vnni, unsigned int M,
                    unsigned int N, ValueRange acc, Value matA, Value matB,
                    unsigned int dimCount) {

  SmallVector<Value> accumulators;
  SmallVector<Value> matLoad;
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

  // Start with assumption that M tile size is smaller and create  the
  // helper variables
  unsigned int outerBound = M;
  unsigned int outerStep = 1;

  unsigned int innerBound = N;
  unsigned int innerStep = vectorSize;

  Value outerMatrix = matA;
  Value innerMatrix = matB;

  unsigned int outerVectSize = vnni;
  unsigned int innerVectSize = vectorSize;

  unsigned int fmaBound = M;

  // update helper variables if N tile size is smaller
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

  // Load all the element of A or B matrix
  for (unsigned int i = 0; i < outerBound; i = i + outerStep) {
    Value indexOp_i = arith::ConstantIndexOp::create(rewriter, loc, i);
    Value valueRow;

    if ((N / vectorSize) > M) {

      // With the assumption as batch-reduce matmul initialize reduction, M, and
      // K dimension.
      SmallVector<Value> index = {c0, indexOp_i, c0};

      // Remove reduction dimension if it is a batch matmul
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      // A Matrix load + broadcast
      Value row = vector::LoadOp::create(
          rewriter, loc, VectorType::get(outerVectSize, elementType),
          outerMatrix, index);
      valueRow = vector::BroadcastOp::create(
          rewriter, loc, VectorType::get(vectorSize, rewriter.getF32Type()),
          row);
    } else {

      // With the assumption as batch-reduce matmul initialize reduction, K, and
      // N dimension.
      SmallVector<Value> index = {c0, c0, indexOp_i};

      // Remove reduction dimension if it is a batch matmul
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      // B Matrix load.
      valueRow = vector::LoadOp::create(
          rewriter, loc, VectorType::get(outerVectSize, elementType),
          outerMatrix, index);
    }

    matLoad.push_back(valueRow);
  }

  // Load elements of A/B Matrix one at a time and compute FMA
  for (unsigned int j = 0, k = 0; j < innerBound; j = j + innerStep) {
    Value indexOp_j = arith::ConstantIndexOp::create(rewriter, loc, j);
    Value valueRow;

    if ((N / vectorSize) < M) {
      SmallVector<Value> index = {c0, indexOp_j, c0};
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      // A Matrix load + broadcast
      Value row = vector::LoadOp::create(
          rewriter, loc, VectorType::get(innerVectSize, elementType),
          innerMatrix, ValueRange(index));
      valueRow = vector::BroadcastOp::create(
          rewriter, loc, VectorType::get(vectorSize, rewriter.getF32Type()),
          row);
    } else {

      SmallVector<Value> index = {c0, c0, indexOp_j};
      if (dimCount == 3) {
        index.erase(index.begin());
      }

      // B Matrix load
      valueRow = vector::LoadOp::create(
          rewriter, loc, VectorType::get(innerVectSize, elementType),
          innerMatrix, index);
    }

    // FMAs
    for (unsigned int i = 0; i < fmaBound; i = i + 1) {
      auto fmaOdd =
          vector::FMAOp::create(rewriter, loc, matLoad[i], valueRow, acc[k]);
      k++;
      accumulators.push_back(fmaOdd);
    }
  }

  return accumulators;
}

// Function to re-create K dimension loop with accumulator as IterArgs for
// lowering a batch-reduce vector contraction to a system specific nanokernels.
scf::ForOp createGEMMLoopsWithAccAsIterArgs(
    RewriterBase &rewriter, Location loc, scf::ForOp kForOp,
    vector::TransferReadOp vectorReadOpLhs,
    vector::TransferReadOp vectorReadOpRhs, Value ivNewReductionForOp,
    Type elementType, unsigned int vectorSize, unsigned int vnni,
    unsigned int M, unsigned int N, ValueRange iterArgsNewReductionForOp,
    unsigned int dimCount) {
  auto newKForOp = scf::ForOp::create(
      rewriter, kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
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

        auto evenFMAs = generateNanokernels(
            rewriter, kForOp.getLoc(), elementType, vectorSize, vnni, M, N,
            iterArgsNewKForOp, lhsClone->getResult(0), rhsClone->getResult(0),
            dimCount);

        scf::YieldOp::create(rewriterNewKForOp, locNewKForOp, evenFMAs);
      });

  return newKForOp;
}

// Function to re-create K dimension loop with accumulator as IterArgs for
// lowering a batch vector contraction to a system specific nanokernels.
scf::ForOp createGEMMLoopsWithAccAsIterArgs(
    RewriterBase &rewriter, Location loc, scf::ForOp kForOp,
    vector::TransferReadOp vectorReadOpLhs,
    vector::TransferReadOp vectorReadOpRhs, Type elementType,
    unsigned int vectorSize, unsigned int vnni, unsigned int M, unsigned int N,
    ValueRange iterArgsNewReductionForOp, unsigned int dimCount) {

  auto newKForOp = scf::ForOp::create(
      rewriter, kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
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
            generateNanokernels(rewriter, loc, elementType, vectorSize, vnni, M,
                                N, iterArgsNewKForOp, lhsClone->getResult(0),
                                rhsClone->getResult(0), dimCount);

        scf::YieldOp::create(rewriterNewKForOp, locNewKForOp, evenFMAs);
      });

  return newKForOp;
}

Value mergeAccumulatedVectorAsMatrix(RewriterBase &rewriter, Location loc,
                                     VectorType vecType,
                                     SmallVector<Value> FMAs, Value accVec,
                                     unsigned int vecSize, unsigned int M,
                                     unsigned int N) {

  auto strides = rewriter.getI64ArrayAttr({1});
  if ((N / vecSize) > M) {
    for (unsigned int j = 0, k = 0; j < (N / vecSize); j++) {
      for (unsigned int i = 0; i < M; i++) {
        unsigned int off = (j * vecSize) + (i * N);
        auto offsets = rewriter.getI64ArrayAttr({off});
        accVec = vector::InsertStridedSliceOp::create(
            rewriter, loc, vecType, FMAs[k], accVec, offsets, strides);
        k++;
      }
    }

  } else {
    for (unsigned int i = 0, k = 0; i < M * N; i = i + vecSize) {
      auto offsets = rewriter.getI64ArrayAttr({i});
      accVec = vector::InsertStridedSliceOp::create(
          rewriter, loc, vecType, FMAs[k], accVec, offsets, strides);
      k++;
    }
  }
  return accVec;
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
    SmallVector<Value> accumulators = loadAccumulatorBeforeGEMM(
        loc, rewriter, elementType, M, N, vectorSize, subviewOpAcc);

    // Create the batch-reduce and K-loop with acc vectors as the loop
    // iterargs (batch-reduce matmul) + nanokernel generation
    scf::ForOp newLoop;
    if (dimCount == 4) {
      newLoop = scf::ForOp::create(
          rewriter, reductionForOp.getLoc(), reductionForOp.getLowerBound(),
          reductionForOp.getUpperBound(), reductionForOp.getStep(),
          accumulators,
          [&](OpBuilder &rewriterNewReductionForOp,
              Location locNewReductionForOp, Value ivNewReductionForOp,
              ValueRange iterArgsNewReductionForOp) {
            scf::ForOp newKForOp = createGEMMLoopsWithAccAsIterArgs(
                rewriter, loc, kForOp, vectorReadOpLhs, vectorReadOpRhs,
                ivNewReductionForOp, elementType, vectorSize, vnni, M, N,
                iterArgsNewReductionForOp, dimCount);

            scf::YieldOp::create(rewriterNewReductionForOp,
                                 locNewReductionForOp, newKForOp.getResults());
          });
    }

    // Create only the K-loop (batch matmul) + nanokernel generation
    if (dimCount == 3) {
      newLoop = createGEMMLoopsWithAccAsIterArgs(
          rewriter, loc, kForOp, vectorReadOpLhs, vectorReadOpRhs, elementType,
          vectorSize, vnni, M, N, accumulators, dimCount);
    }

    // Combine all acc vectors into a MxN C matrix
    auto vecType = VectorType::get({M * N}, rewriter.getF32Type());
    auto zeroAttr =
        DenseElementsAttr::get(vecType, rewriter.getF32FloatAttr(0.0));
    Value accVec = arith::ConstantOp::create(rewriter, loc, vecType, zeroAttr);

    accVec = mergeAccumulatedVectorAsMatrix(
        rewriter, loc, vecType, newLoop.getResults(), accVec, vectorSize, M, N);

    auto accTy = dyn_cast<VectorType>(contractOp.getAccType());
    auto reshapeAcc = vector::ShapeCastOp::create(rewriter, loc, accTy, accVec);

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
