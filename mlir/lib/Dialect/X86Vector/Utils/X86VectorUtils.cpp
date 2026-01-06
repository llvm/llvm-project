//===- X86VectorUtils.cpp - MLIR Utilities for X86VectorOps   -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86Vector/Utils/X86VectorUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include "llvm/ADT/ArrayRef.h"
#include <cassert>

namespace mlir {
namespace x86vector {

static FailureOr<SmallVector<mlir::utils::IteratorType>>
inferIteratorsFromOutMap(AffineMap map) {
  if (!map.isProjectedPermutation())
    return failure();
  SmallVector<mlir::utils::IteratorType> iterators(
      map.getNumDims(), mlir::utils::IteratorType::reduction);
  for (auto expr : map.getResults())
    if (auto dim = dyn_cast<AffineDimExpr>(expr))
      iterators[dim.getPosition()] = mlir::utils::IteratorType::parallel;
  return iterators;
}

// Returns true if the operation is in VNNI layout.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(Operation *op, ArrayRef<AffineMap> indexingMaps,
                    std::optional<unsigned> blockingFactor) {
  // Narrow down type operations - VNNI only applies to contractions.
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(indexingMaps);
  if (failed(dims))
    return false;

  auto matA = op->getOperand(0);
  auto matB = op->getOperand(1);
  auto typeA = dyn_cast<ShapedType>(matA.getType());
  auto typeB = dyn_cast<ShapedType>(matB.getType());
  unsigned rankA = typeA.getRank();
  unsigned rankB = typeB.getRank();
  // VNNI format requires at least 1 parallel and 2 reduction dimensions.
  if (rankA < 3 || rankB < 3)
    return false;

  // At least two reduction dimensions are expected:
  // one for the VNNI factor and one for the K dimension
  if (dims->k.size() < 2)
    return false;

  // Validate affine maps - VNNI computation should be defined by the two
  // innermost reduction iterators.
  // The input matrix dimensions layout must match the following:
  //   - matrix A - [...][K/vnniFactor][vnniFactor]
  //   - matrix B - [...][K/vnniFactor][N][vnniFactor]
  auto maybeIters = inferIteratorsFromOutMap(indexingMaps[2] /* outs */);
  if (failed(maybeIters))
    return false;
  SmallVector<mlir::utils::IteratorType> iteratorTypes = *maybeIters;
  AffineMap mapA = indexingMaps[0];
  AffineMap mapB = indexingMaps[1];

  auto vnniDimA = dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1));
  auto vnniDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 1));
  if (!vnniDimA || !vnniDimB || vnniDimA != vnniDimB ||
      iteratorTypes[vnniDimA.getPosition()] !=
          mlir::utils::IteratorType::reduction)
    return false;
  auto redDimA = dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2));
  auto redDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 3));
  if (!redDimA || !redDimB || redDimA != redDimB ||
      iteratorTypes[redDimA.getPosition()] !=
          mlir::utils::IteratorType::reduction)
    return false;
  auto parallelDimB = dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 2));
  if (!parallelDimB || iteratorTypes[parallelDimB.getPosition()] !=
                           mlir::utils::IteratorType::parallel)
    return false;

  // VNNI factor must be:
  //   - the innermost inputs' dimension
  //   - statically known
  //   - multiple of 2 or equal to the specified factor
  auto vnniDimSize = typeB.getShape().back();
  if (vnniDimSize == ShapedType::kDynamic || vnniDimSize == 0 ||
      vnniDimSize % 2 != 0)
    return false;
  if (typeA.getShape().back() != vnniDimSize)
    return false;
  if (blockingFactor && vnniDimSize != *blockingFactor)
    return false;

  // The split reduction dimension size should also match.
  if (typeA.getShape().end()[-2] != typeB.getShape().end()[-3])
    return false;

  return true;
}

struct ShuffleMasks {
  llvm::ArrayRef<int64_t> maskLo;
  llvm::ArrayRef<int64_t> maskHi;
};

inline ShuffleMasks getShuffleMasks(int64_t nonUnitDimAcc) {
  // We only support these two layouts for now.
  assert((nonUnitDimAcc == 8 || nonUnitDimAcc == 16) &&
         "Unsupported nonUnitDimAcc value");

  static constexpr int64_t maskLo8[] = {0, 8, 1, 9, 2, 10, 3, 11};
  static constexpr int64_t maskHi8[] = {4, 12, 5, 13, 6, 14, 7, 15};

  static constexpr int64_t maskLo16[] = {0, 1, 2, 3, 16, 17, 18, 19,
                                         4, 5, 6, 7, 20, 21, 22, 23};
  static constexpr int64_t maskHi16[] = {8,  9,  10, 11, 24, 25, 26, 27,
                                         12, 13, 14, 15, 28, 29, 30, 31};

  if (nonUnitDimAcc == 16)
    return {maskLo16, maskHi16};

  // nonUnitDimAcc == 8
  return {maskLo8, maskHi8};
}

Operation *traceToVectorReadLikeParentOperation(Value v) {
  while (true) {
    // Case 1: Value defined by an operation
    if (Operation *defOp = v.getDefiningOp()) {
      if (isa<vector::TransferReadOp, vector::LoadOp>(defOp)) {
        return defOp;
      }

      if (isa<vector::ShapeCastOp, vector::ShuffleOp>(defOp)) {
        return nullptr;
      }

      // Continue tracing (accumulators usually forward the value)
      if (defOp->getNumOperands() == 1) {
        v = defOp->getOperand(0);
        continue;
      }

      return nullptr;
    }

    // Case 2: BlockArgument (scf.for iter_arg)
    if (auto barg = dyn_cast<BlockArgument>(v)) {
      auto *parentOp = barg.getOwner()->getParentOp();

      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned argNum = barg.getArgNumber();

        // arg0 = induction variable (not an iter_arg)
        if (argNum == 0)
          return nullptr;

        unsigned iterIdx = argNum - 1;
        v = forOp.getInitArgs()[iterIdx];
        continue;
      }

      return nullptr;
    }

    return nullptr;
  }
}

Operation *traceToVectorWriteLikeUserOperation(Value v) {
  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();

    // --- TERMINAL OPS ---
    if (isa<vector::TransferWriteOp>(user) || isa<vector::StoreOp>(user)) {
      return user;
    }

    if (isa<vector::ShapeCastOp, vector::ShuffleOp>(user)) {
      return nullptr;
    }

    // --- SCF YIELD ---
    if (auto yield = dyn_cast<scf::YieldOp>(user)) {
      Operation *parent = yield->getParentOp();
      unsigned idx = use.getOperandNumber();
      if (auto *res =
              traceToVectorWriteLikeUserOperation(parent->getResult(idx)))
        return res;
      continue;
    }

    // --- SCF FOR ---
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      unsigned idx = use.getOperandNumber();
      if (auto *res = traceToVectorWriteLikeUserOperation(forOp.getResult(idx)))
        return res;
      continue;
    }

    // --- GENERIC CASE ---
    for (Value res : user->getResults()) {
      if (auto *found = traceToVectorWriteLikeUserOperation(res))
        return found;
    }
  }

  return nullptr;
}

static void rewriteUses(mlir::Value oldVal, mlir::Value newVal,
                        mlir::Operation *targetContract,
                        mlir::PatternRewriter &rewriter) {
  for (mlir::OpOperand &use : llvm::make_early_inc_range(oldVal.getUses())) {

    mlir::Operation *user = use.getOwner();

    // if (user == targetContract ||
    if (mlir::isa<mlir::vector::ContractionOp>(user) ||
        mlir::isa<mlir::scf::ForOp>(user)) {
      use.set(newVal);
    }
  }
}

void shuffleAfterReadLikeOp(mlir::PatternRewriter &rewriter,
                            mlir::Operation *opA, mlir::Operation *opB,
                            mlir::vector::ContractionOp contractA,
                            mlir::vector::ContractionOp contractB,
                            int64_t nonUnitDimAcc, mlir::VectorType accTy) {
  mlir::Operation *insertAfter = opA->isBeforeInBlock(opB) ? opB : opA;

  rewriter.setInsertionPointAfter(insertAfter);
  mlir::Location loc = insertAfter->getLoc();

  auto elemTy = accTy.getElementType();
  auto flatTy = mlir::VectorType::get(nonUnitDimAcc, elemTy);

  auto castA = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                 opA->getResult(0));

  auto castB = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                 opB->getResult(0));

  auto masks = getShuffleMasks(nonUnitDimAcc);

  auto shuffleLo = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                                   castB, masks.maskLo);

  auto shuffleHi = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                                   castB, masks.maskHi);

  auto newAccA =
      mlir::vector::ShapeCastOp::create(rewriter, loc, accTy, shuffleLo);

  auto newAccB =
      mlir::vector::ShapeCastOp::create(rewriter, loc, accTy, shuffleHi);

  rewriteUses(opA->getResult(0), newAccA.getResult(), contractA, rewriter);

  rewriteUses(opB->getResult(0), newAccB.getResult(), contractB, rewriter);
}

void shuffleBeforeWriteLikeOp(mlir::PatternRewriter &rewriter,
                              mlir::Operation *opA, mlir::Operation *opB,
                              int64_t nonUnitDimAcc, mlir::VectorType accTy) {
  // Helper to extract vector operand from write-like ops
  auto getWrittenVector = [](mlir::Operation *op) -> mlir::Value {
    if (auto write = mlir::dyn_cast<mlir::vector::TransferWriteOp>(op))
      return write.getVector();
    if (auto store = mlir::dyn_cast<mlir::vector::StoreOp>(op))
      return store.getValueToStore();
    return nullptr;
  };

  mlir::Value vecA = getWrittenVector(opA);
  mlir::Value vecB = getWrittenVector(opB);

  assert(vecA && vecB && "expected vector write-like ops");

  // Decide insertion point and location
  mlir::Operation *insertBefore = opA->isBeforeInBlock(opB) ? opA : opB;

  rewriter.setInsertionPoint(insertBefore);
  mlir::Location loc = insertBefore->getLoc();

  auto elemTy = accTy.getElementType();
  auto flatTy = mlir::VectorType::get(nonUnitDimAcc, elemTy);

  // Flatten vectors
  auto castA = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy, vecA);

  auto castB = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy, vecB);

  // TODO: derive shuffle masks instead of hard-coding
  auto masks = getShuffleMasks(nonUnitDimAcc);

  auto shuffledLo = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy,
                                                    castA, castB, masks.maskLo);

  auto shuffledHi = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy,
                                                    castA, castB, masks.maskHi);

  // Cast back to accumulator type
  auto newVecA =
      mlir::vector::ShapeCastOp::create(rewriter, loc, accTy, shuffledLo);

  auto newVecB =
      mlir::vector::ShapeCastOp::create(rewriter, loc, accTy, shuffledHi);

  // Update write operands in place
  opA->setOperand(0, newVecA.getResult());
  opB->setOperand(0, newVecB.getResult());
}

void shuffleNonUnitDimOperand(mlir::PatternRewriter &rewriter,
                              mlir::Operation *opA, mlir::Operation *opB,
                              mlir::vector::ContractionOp contractA,
                              mlir::vector::ContractionOp contractB,
                              int64_t nonUnitDimAcc, mlir::VectorType Ty) {
  mlir::Operation *insertAfter = opA->isBeforeInBlock(opB) ? opB : opA;

  rewriter.setInsertionPointAfter(insertAfter);
  mlir::Location loc = insertAfter->getLoc();

  auto elemTy = Ty.getElementType();
  auto flatTy = mlir::VectorType::get(nonUnitDimAcc, elemTy);

  auto castA = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                 opA->getResult(0));

  auto castB = mlir::vector::ShapeCastOp::create(rewriter, loc, flatTy,
                                                 opB->getResult(0));

  static constexpr int64_t maskLo[] = {
      0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,  41, 10, 42, 11, 43,
      16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59};
  static constexpr int64_t maskHi[] = {
      4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13, 45, 14, 46, 15, 47,
      20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63};

  auto shuffleLo = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                                   castB, maskLo);

  auto shuffleHi = mlir::vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                                   castB, maskHi);

  auto newAccA =
      mlir::vector::ShapeCastOp::create(rewriter, loc, Ty, shuffleLo);

  auto newAccB =
      mlir::vector::ShapeCastOp::create(rewriter, loc, Ty, shuffleHi);

  rewriteUses(opA->getResult(0), newAccA.getResult(), contractA, rewriter);

  rewriteUses(opB->getResult(0), newAccB.getResult(), contractB, rewriter);
}

bool validatePairVectorContract(vector::ContractionOp contractOp,
                                vector::ContractionOp pairContOp,
                                bool rhsHasMultipleNonUnitDims,
                                int64_t nonUnitDimValue) {

  if (!(contractOp.getLhs() == pairContOp.getLhs()) &&
      !(contractOp.getRhs() == pairContOp.getRhs()))
    return false;

  if (rhsHasMultipleNonUnitDims &&
      !(contractOp.getLhs() == pairContOp.getLhs()))
    return false;

  if (!rhsHasMultipleNonUnitDims &&
      !(contractOp.getRhs() == pairContOp.getRhs()))
    return false;

  auto op =
      rhsHasMultipleNonUnitDims ? contractOp.getRhs() : contractOp.getLhs();
  auto op1 =
      rhsHasMultipleNonUnitDims ? pairContOp.getRhs() : pairContOp.getLhs();

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<mlir::Operation *>(op.getDefiningOp())
      .Case<vector::TransferReadOp, vector::LoadOp>([&](auto readOp) {
        srcBuff = readOp.getOperand(0);
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
      });

  Value srcBuff1;
  SmallVector<OpFoldResult> indexVals1;
  llvm::TypeSwitch<mlir::Operation *>(op1.getDefiningOp())
      .Case<vector::TransferReadOp, vector::LoadOp>([&](auto readOp) {
        srcBuff1 = readOp.getOperand(0);
        indexVals1 = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                               readOp.getIndices().end());
      });

  if (!srcBuff || !srcBuff1)
    return false;

  if (!(srcBuff == srcBuff1))
    return false;

  for (size_t i = 0; i < indexVals.size(); i++) {
    if (getConstantIntValue(indexVals[i]) == getConstantIntValue(indexVals1[i]))
      continue;

    auto value1 = *getConstantIntValue(indexVals1[i]);
    auto value2 = *getConstantIntValue(indexVals[i]);

    if ((value1 - value2) != nonUnitDimValue)
      return false;
  }

  return true;
}

} // namespace x86vector
} // namespace mlir
