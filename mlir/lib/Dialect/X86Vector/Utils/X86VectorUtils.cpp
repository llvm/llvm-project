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
  // Do interleaving between two <8xf32> targeting AVX2.
  static constexpr int64_t maskLo8[] = {0, 8, 1, 9, 2, 10, 3, 11};
  static constexpr int64_t maskHi8[] = {4, 12, 5, 13, 6, 14, 7, 15};

  // Shuffle two <16xf32> as below targeting AVX512.
  static constexpr int64_t maskLo16[] = {0, 1, 2, 3, 16, 17, 18, 19,
                                         4, 5, 6, 7, 20, 21, 22, 23};
  static constexpr int64_t maskHi16[] = {8,  9,  10, 11, 24, 25, 26, 27,
                                         12, 13, 14, 15, 28, 29, 30, 31};

  if (nonUnitDimAcc == 16)
    return {maskLo16, maskHi16};

  return {maskLo8, maskHi8};
}

// This function walks backward from a value to locate its originating
// vector read-like operation (`vector.transfer_read` or `vector.load`).
// It follows simple forwarding through unary ops and across `scf.for`
// loop iter-arguments, while stopping if layout-transforming ops such
// as `shape_cast` or `shuffle` are encountered. The traversal returns
// the read-like defining operation or `nullptr` if no valid source
// is found.
Operation *traceToVectorReadLikeParentOperation(Value v) {
  while (true) {
    // Case 1: Value defined by an operation
    if (Operation *defOp = v.getDefiningOp()) {
      if (isa<vector::TransferReadOp, vector::LoadOp>(defOp))
        return defOp;

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

// This function recursively traces a value through its uses to find
// a downstream vector write-like operation (`vector.transfer_write`
// or `vector.store`). It transparently follows values across `scf.for`
// and `scf.yield` boundaries while stopping if layout-altering ops such
// as `shape_cast` or `shuffle` are encountered. The traversal returns
// the  matching write-like user. Returns `nullptr` if none is found or
// the value has multiple users.
Operation *traceToVectorWriteLikeUserOperation(Value v) {

  if (v.getNumUses() > 1)
    return nullptr;

  for (OpOperand &use : v.getUses()) {
    Operation *user = use.getOwner();

    // --- TERMINAL OPS ---
    if (isa<vector::TransferWriteOp>(user) || isa<vector::StoreOp>(user))
      return user;

    if (isa<vector::ShapeCastOp, vector::ShuffleOp>(user))
      return nullptr;

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

// This function packs the accumulator of two flat BF16 vector.contract
// operations into VNNI packed and are then replaced in their respective
// contraction ops, enabling post-read layout or packing transformations.
// TODO: replace all use with the packed value along with contration
// and for op.
LogicalResult shuffleAfterReadLikeOp(PatternRewriter &rewriter, Operation *opA,
                                     Operation *opB,
                                     vector::ContractionOp contractA,
                                     vector::ContractionOp contractB,
                                     int64_t nonUnitDimAcc, VectorType accTy) {

  if (!isa<vector::TransferReadOp, vector::LoadOp>(opA) ||
      !isa<vector::TransferReadOp, vector::LoadOp>(opB)) {
    return failure();
  }

  Operation *insertAfter = opA->isBeforeInBlock(opB) ? opB : opA;

  rewriter.setInsertionPointAfter(insertAfter);
  Location loc = insertAfter->getLoc();

  auto elemTy = accTy.getElementType();
  auto flatTy = VectorType::get(nonUnitDimAcc, elemTy);

  auto castA =
      vector::ShapeCastOp::create(rewriter, loc, flatTy, opA->getResult(0));
  auto castB =
      vector::ShapeCastOp::create(rewriter, loc, flatTy, opB->getResult(0));

  auto masks = getShuffleMasks(nonUnitDimAcc);

  auto shuffleLo = vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                             castB, masks.maskLo);
  auto shuffleHi = vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                             castB, masks.maskHi);

  auto newAccA = vector::ShapeCastOp::create(rewriter, loc, accTy, shuffleLo);
  auto newAccB = vector::ShapeCastOp::create(rewriter, loc, accTy, shuffleHi);

  rewriter.replaceUsesWithIf(
      opA->getResult(0), newAccA.getResult(), [&](OpOperand &use) {
        return isa<vector::ContractionOp, scf::ForOp>(use.getOwner());
      });

  rewriter.replaceUsesWithIf(
      opB->getResult(0), newAccB.getResult(), [&](OpOperand &use) {
        return isa<vector::ContractionOp, scf::ForOp>(use.getOwner());
      });

  return success();
}

// This function shuffles the vectors written by vector.contract operation
// as a flat layout structure before they are stored.
LogicalResult shuffleBeforeWriteLikeOp(PatternRewriter &rewriter,
                                       Operation *opA, Operation *opB,
                                       int64_t nonUnitDimAcc,
                                       VectorType accTy) {
  // Helper to extract vector operand from write-like ops
  auto getWrittenVector = [](Operation *op) -> Value {
    if (auto write = dyn_cast<vector::TransferWriteOp>(op))
      return write.getVector();
    if (auto store = dyn_cast<vector::StoreOp>(op))
      return store.getValueToStore();
    return nullptr;
  };

  Value vecA = getWrittenVector(opA);
  Value vecB = getWrittenVector(opB);

  if (!vecA || !vecB)
    return failure();

  // Decide insertion point and location
  Operation *insertBefore = opA->isBeforeInBlock(opB) ? opA : opB;

  rewriter.setInsertionPoint(insertBefore);
  Location loc = insertBefore->getLoc();

  auto elemTy = accTy.getElementType();
  auto flatTy = VectorType::get(nonUnitDimAcc, elemTy);

  // Flatten vectors
  auto castA = vector::ShapeCastOp::create(rewriter, loc, flatTy, vecA);
  auto castB = vector::ShapeCastOp::create(rewriter, loc, flatTy, vecB);

  // TODO: derive shuffle masks instead of hard-coding
  auto masks = getShuffleMasks(nonUnitDimAcc);

  auto shuffledLo = vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                              castB, masks.maskLo);
  auto shuffledHi = vector::ShuffleOp::create(rewriter, loc, flatTy, castA,
                                              castB, masks.maskHi);

  // Cast back to accumulator type
  auto newVecA = vector::ShapeCastOp::create(rewriter, loc, accTy, shuffledLo);
  auto newVecB = vector::ShapeCastOp::create(rewriter, loc, accTy, shuffledHi);

  // Update write operands in place
  opA->setOperand(0, newVecA.getResult());
  opB->setOperand(0, newVecB.getResult());

  return success();
}

// Return true if vector.contract operations matches on below conditions:
//  (1) - the unitDim operand Lhs or Rhs should be same,
//  (2) - the defining source memref should be same for nonUnitDim
//  operation,
//  (3) - the nonUnit dim offset difference between the
//  vector.contracts should be 8 or 16.
bool validatePairVectorContract(vector::ContractionOp contractOp,
                                vector::ContractionOp pairContOp,
                                bool rhsHasMultipleNonUnitDims,
                                int64_t nonUnitDimValue) {
  if (rhsHasMultipleNonUnitDims &&
      !(contractOp.getLhs() == pairContOp.getLhs()))
    return false;

  if (!rhsHasMultipleNonUnitDims &&
      !(contractOp.getRhs() == pairContOp.getRhs()))
    return false;

  auto nonUnitOperand =
      rhsHasMultipleNonUnitDims ? contractOp.getRhs() : contractOp.getLhs();
  auto nonUnitOperandPairContOp =
      rhsHasMultipleNonUnitDims ? pairContOp.getRhs() : pairContOp.getLhs();

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(nonUnitOperand.getDefiningOp())
      .Case<vector::TransferReadOp, vector::LoadOp>([&](auto readOp) {
        srcBuff = readOp.getOperand(0);
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
      })
      .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp op) {
        srcBuff = op.getSource();
        indexVals.clear();
      });

  Value srcBuffPairContOp;
  SmallVector<OpFoldResult> indexValsPairContOp;
  llvm::TypeSwitch<Operation *>(nonUnitOperandPairContOp.getDefiningOp())
      .Case<vector::TransferReadOp, vector::LoadOp>([&](auto readOp) {
        srcBuffPairContOp = readOp.getOperand(0);
        indexValsPairContOp = SmallVector<OpFoldResult>(
            readOp.getIndices().begin(), readOp.getIndices().end());
      })
      .Case<vector::ShapeCastOp>([&](vector::ShapeCastOp op) {
        srcBuffPairContOp = op.getSource();
        indexVals.clear();
      });

  if (!srcBuff || !srcBuffPairContOp)
    return false;

  auto shuffleLw = srcBuff.getDefiningOp<vector::ShuffleOp>();
  auto shuffleHw = srcBuffPairContOp.getDefiningOp<vector::ShuffleOp>();

  if (shuffleLw && shuffleHw)
    return shuffleLw.getV1() == shuffleHw.getV1() &&
           shuffleLw.getV2() == shuffleHw.getV2();

  if (srcBuff != srcBuffPairContOp)
    return false;

  for (size_t i = 0; i < indexVals.size(); i++) {
    auto v0 = getConstantIntValue(indexVals[i]);
    auto v1 = getConstantIntValue(indexValsPairContOp[i]);

    if (!v0 || !v1)
      return false;

    if (*v1 == *v0)
      continue;

    if ((*v1 - *v0) != nonUnitDimValue)
      return false;
  }

  return true;
}

} // namespace x86vector
} // namespace mlir
