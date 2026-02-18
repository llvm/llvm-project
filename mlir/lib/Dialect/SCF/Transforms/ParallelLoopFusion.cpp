//===- ParallelLoopFusion.cpp - Code to perform loop fusion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop fusion on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>
#include <tuple>

namespace mlir {
#define GEN_PASS_DEF_SCFPARALLELLOOPFUSION
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// Verify there are no nested ParallelOps.
static bool hasNestedParallelOp(ParallelOp ploop) {
  auto walkResult =
      ploop.getBody()->walk([](ParallelOp) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

/// Verify equal iteration spaces.
static bool equalIterationSpaces(ParallelOp firstPloop,
                                 ParallelOp secondPloop) {
  if (firstPloop.getNumLoops() != secondPloop.getNumLoops())
    return false;

  auto matchOperands = [&](const OperandRange &lhs,
                           const OperandRange &rhs) -> bool {
    // TODO: Extend this to support aliases and equal constants.
    return std::equal(lhs.begin(), lhs.end(), rhs.begin());
  };
  return matchOperands(firstPloop.getLowerBound(),
                       secondPloop.getLowerBound()) &&
         matchOperands(firstPloop.getUpperBound(),
                       secondPloop.getUpperBound()) &&
         matchOperands(firstPloop.getStep(), secondPloop.getStep());
}

/// Check if both operations are the same type of memory write op and
/// write to the same memory location (same buffer and same indices).
static bool opsWriteSameMemLocation(Operation *op1, Operation *op2) {
  if (!op1 || !op2 || op1->getName() != op2->getName())
    return false;
  if (op1 == op2)
    return true;
  // support only these memory-writing ops for now
  if (!isa<memref::StoreOp, vector::TransferWriteOp, vector::StoreOp>(op1))
    return false;
  bool opsAreIdentical =
      llvm::TypeSwitch<Operation *, bool>(op1)
          .Case([&](memref::StoreOp storeOp1) {
            auto storeOp2 = cast<memref::StoreOp>(op2);
            return (storeOp1.getMemRef() == storeOp2.getMemRef()) &&
                   (storeOp1.getIndices() == storeOp2.getIndices());
          })
          .Case([&](vector::TransferWriteOp writeOp1) {
            auto writeOp2 = cast<vector::TransferWriteOp>(op2);
            return (writeOp1.getBase() == writeOp2.getBase()) &&
                   (writeOp1.getIndices() == writeOp2.getIndices()) &&
                   (writeOp1.getMask() == writeOp2.getMask()) &&
                   (writeOp1.getValueToStore().getType() ==
                    writeOp2.getValueToStore().getType()) &&
                   (writeOp1.getInBounds() == writeOp2.getInBounds());
          })
          .Case([&](vector::StoreOp vecStoreOp1) {
            auto vecStoreOp2 = cast<vector::StoreOp>(op2);
            return (vecStoreOp1.getBase() == vecStoreOp2.getBase()) &&
                   (vecStoreOp1.getIndices() == vecStoreOp2.getIndices()) &&
                   (vecStoreOp1.getValueToStore().getType() ==
                    vecStoreOp2.getValueToStore().getType()) &&
                   (vecStoreOp1.getAlignment() == vecStoreOp2.getAlignment()) &&
                   (vecStoreOp1.getNontemporal() ==
                    vecStoreOp2.getNontemporal());
          })
          .Default([](Operation *) { return false; });
  return opsAreIdentical;
}

/// Check if val1 (from the first parallel loop) and val2 (from the
/// second) are equivalent, considering the mapping of induction variables from
/// the first to the second parallel loop.
static bool valsAreEquivalent(Value val1, Value val2,
                              const IRMapping &loopsIVsMap) {
  if (val1 == val2 || loopsIVsMap.lookupOrDefault(val1) == val2 ||
      loopsIVsMap.lookupOrDefault(val2) == val1)
    return true;
  Operation *val1DefOp = val1.getDefiningOp();
  Operation *val2DefOp = val2.getDefiningOp();
  if (!val1DefOp || !val2DefOp)
    return false;
  if (!isMemoryEffectFree(val1DefOp) || !isMemoryEffectFree(val2DefOp))
    return false;
  return OperationEquivalence::isEquivalentTo(
      val1DefOp, val2DefOp,
      [&](Value v1, Value v2) {
        return success(loopsIVsMap.lookupOrDefault(v1) == v2 ||
                       loopsIVsMap.lookupOrDefault(v2) == v1);
      },
      /*markEquivalent=*/nullptr, OperationEquivalence::Flags::IgnoreLocations);
}

/// If the `expr` value is the result of an integer addition of `base` and a
/// constant, return the constant.
static std::optional<int64_t> getAddConstant(Value expr, Value base,
                                             const IRMapping &loopsIVsMap) {
  if (auto addOp = expr.getDefiningOp<arith::AddIOp>()) {
    if (auto constOp = getConstantIntValue(addOp.getLhs());
        constOp && valsAreEquivalent(addOp.getRhs(), base, loopsIVsMap))
      return constOp.value();
    if (auto constOp = getConstantIntValue(addOp.getRhs());
        constOp && valsAreEquivalent(addOp.getLhs(), base, loopsIVsMap))
      return constOp.value();
    return std::nullopt;
  }

  if (auto addOp = expr.getDefiningOp<index::AddOp>()) {
    if (auto constOp = getConstantIntValue(addOp.getLhs());
        constOp && valsAreEquivalent(addOp.getRhs(), base, loopsIVsMap))
      return constOp.value();
    if (auto constOp = getConstantIntValue(addOp.getRhs());
        constOp && valsAreEquivalent(addOp.getLhs(), base, loopsIVsMap))
      return constOp.value();
    return std::nullopt;
  }

  if (auto applyOp = expr.getDefiningOp<affine::AffineApplyOp>()) {
    AffineMap map = applyOp.getAffineMap();
    if (map.getNumResults() != 1 || map.getNumDims() != 1 ||
        map.getNumSymbols() != 0)
      return std::nullopt;
    if (!valsAreEquivalent(applyOp.getOperand(0), base, loopsIVsMap))
      return std::nullopt;
    AffineExpr result = map.getResult(0);
    auto bin = dyn_cast<AffineBinaryOpExpr>(result);
    if (!bin || bin.getKind() != AffineExprKind::Add)
      return std::nullopt;
    auto lhsDim = dyn_cast<AffineDimExpr>(bin.getLHS());
    auto rhsDim = dyn_cast<AffineDimExpr>(bin.getRHS());
    auto lhsConst = dyn_cast<AffineConstantExpr>(bin.getLHS());
    auto rhsConst = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (lhsConst && rhsDim)
      return lhsConst.getValue();
    if (rhsConst && lhsDim)
      return rhsConst.getValue();
  }
  return std::nullopt;
}

// Return true if the scalar load index may hit any element covered by a
// vector.store/transfer_write along a single memref dimension. Supported cases:
//
// 1) Direct index match (with optional offset):
//    vector.transfer_write %v, %A[%i] : vector<4xf32>, memref<...>
//    %x = memref.load %A[%i] : memref<...>
//
// 2) Loop IV range intersects the write range:
//    vector.transfer_write %v, %A[%c0] : vector<4xf32>, memref<...>
//    scf.for %k = %c0 to %c4 step %c1 { %x = memref.load %A[%k] }
//
// 3) Constant index (or IV + constant) within the write range:
//    vector.transfer_write %v, %A[%c0] : vector<4xf32>, memref<...>
//    %x = memref.load %A[%c2] : memref<...>
//    %y = memref.load %A[%i + %c1] : memref<...>
//
// Args:
// - loadIndex: index used by the scalar load for this dimension.
// - offset: subview offset for the base memref dimension (if any).
// - writeIndex: index used by the transfer_write for this dimension. Can be
// null if the dim was dropped by a rank reducing subview, whose result is
// written by the vector.write.
// - extent: vector size along this dimension (number of elements written).
// - loopsIVsMap: IV equivalence map between fused loops.
static bool loadIndexWithinWriteRange(Value loadIndex, OpFoldResult offset,
                                      Value writeIndex, int64_t extent,
                                      const IRMapping &loopsIVsMap) {
  if (extent <= 0)
    return false;

  // Extract constant loop bounds for loop IVs (e.g. from scf.for).
  auto getConstLoopBoundsForIV =
      [](Value index) -> std::optional<std::tuple<int64_t, int64_t, int64_t>> {
    auto blockArg = dyn_cast<BlockArgument>(index);
    if (!blockArg)
      return std::nullopt;
    auto *parentOp = blockArg.getOwner()->getParentOp();
    auto loopLike = dyn_cast<LoopLikeOpInterface>(parentOp);
    if (!loopLike)
      return std::nullopt;
    auto ranges = getConstLoopBounds(loopLike);
    if (ranges.empty())
      return std::nullopt;

    auto ivs = loopLike.getLoopInductionVars();
    if (!ivs)
      return std::nullopt;
    auto it = llvm::find(*ivs, blockArg);
    if (it == ivs->end())
      return std::nullopt;
    unsigned pos = std::distance(ivs->begin(), it);
    if (pos >= ranges.size())
      return std::nullopt;
    auto [lb, ub, step] = ranges[pos];
    return std::make_tuple(lb, ub, step);
  };

  std::optional<int64_t> offsetConst = getConstantIntValue(offset);
  std::optional<int64_t> writeConst =
      writeIndex ? getConstantIntValue(writeIndex) : std::optional<int64_t>(0);
  if (!writeConst && writeIndex) {
    // Treat single-iteration IVs as constants for matching.
    if (auto bounds = getConstLoopBoundsForIV(writeIndex)) {
      auto [lb, ub, step] = *bounds;
      if (step > 0 && ub == lb + step)
        writeConst = lb;
    }
  }

  // Check whether a loop IV is fully contained in a constant write range.
  auto loopIVWithinRange = [](int64_t lb, int64_t ub, int64_t step,
                              int64_t rangeStart, int64_t rangeExtent) -> bool {
    if (rangeExtent <= 0 || step <= 0)
      return false;
    if (ub <= lb)
      return false;
    int64_t rangeEnd = rangeStart + rangeExtent;
    return lb >= rangeStart && ub <= rangeEnd;
  };

  if (offsetConst && writeConst) {
    // Constant start of the write range; check constant load or loop IV range.
    int64_t start = *offsetConst + *writeConst;
    if (auto loadConst = getConstantIntValue(loadIndex))
      return (*loadConst >= start && *loadConst < start + extent);
    if (auto bounds = getConstLoopBoundsForIV(loadIndex)) {
      auto [lb, ub, step] = *bounds;
      return loopIVWithinRange(lb, ub, step, start, extent);
    }
  }

  if (writeIndex) {
    // Direct IV match (or IV + constant) against the write index.
    if (offsetConst && *offsetConst == 0 &&
        valsAreEquivalent(loadIndex, writeIndex, loopsIVsMap))
      return true;
    if (auto addConst = getAddConstant(loadIndex, writeIndex, loopsIVsMap)) {
      // Match load index of the form writeIndex + C within the write extent.
      if (offsetConst) {
        int64_t start = *offsetConst;
        return (*addConst >= start && *addConst < start + extent);
      }
    }
    return false;
  }

  if (auto offsetVal = dyn_cast<Value>(offset)) {
    // Exact match when extent is 1 and the load hits the offset value.
    if (extent == 1 && valsAreEquivalent(loadIndex, offsetVal, loopsIVsMap))
      return true;
  }

  return false;
}

/// Return the base memref value used by the given memory op.
static Value getBaseMemref(Operation *op) {
  // TODO: use the common interface for memory ops once available.
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case([&](memref::LoadOp load) { return load.getMemRef(); })
      .Case([&](memref::StoreOp store) { return store.getMemRef(); })
      .Case([&](vector::TransferReadOp read) { return read.getBase(); })
      .Case([&](vector::TransferWriteOp write) { return write.getBase(); })
      .Case([&](vector::LoadOp load) { return load.getBase(); })
      .Case([&](vector::StoreOp store) { return store.getBase(); })
      .Default([](Operation *) { return Value(); });
}

/// Recognize scalar memref.load of an element produced by a vector write
/// (vector.transfer_write or vector.store, optionally through a rank-reducing
/// unit-stride subview) of the same buffer. This covers the pattern where a
/// vector write stores a full lane pack and a subsequent scalar load reads an
/// element from that lane pack. EXAMPLE:
///  vector.transfer_write %V, %arg[%x, %y, ..., 0] {in_bounds = [true]} :
///             vector<4xf32>, memref<4xf32, strided<[1], offset: ?>>
///  scf.for %iter = %c0 to %c4 step %c1 iter_args(...) -> (f32) {
///    %0 = memref.load %arg[%x, %y, ..., %iter] : memref<1x128x16x4xf32>
///    ...
///  }
///
static bool isLoadOnWrittenVector(memref::LoadOp loadOp, Value writeBase,
                                  ValueRange writeIndices, VectorType vecTy,
                                  ArrayRef<int64_t> vectorDimForWriteDim,
                                  const IRMapping &ivsMap) {
  if (!vecTy)
    return false;

  Value base = writeBase;
  // The write base if there is no subview, or the subview source otherwise.
  MemrefValue baseMemref = nullptr;
  SmallVector<OpFoldResult> offsets;
  llvm::SmallBitVector droppedDims;
  bool hasSubview = false;
  auto *ctx = loadOp.getContext();
  if (auto subView = base.getDefiningOp<memref::SubViewOp>()) {
    if (!subView.hasUnitStride())
      return false;
    baseMemref = cast<MemrefValue>(subView.getSource());
    offsets = llvm::to_vector(subView.getMixedOffsets());
    droppedDims = subView.getDroppedDims();
    hasSubview = true;
  } else {
    baseMemref = dyn_cast<MemrefValue>(base);
    if (!baseMemref)
      return false;
  }

  auto loadIndices = loadOp.getIndices();
  unsigned baseRank = baseMemref.getType().getRank();
  if ((loadOp.getMemref() != baseMemref) || (loadIndices.size() != baseRank))
    return false;

  unsigned writeRank = writeIndices.size();
  if ((!hasSubview && writeRank != baseRank) ||
      (hasSubview && offsets.size() != baseRank) ||
      (vectorDimForWriteDim.size() != writeRank))
    return false;

  auto zeroAttr = IntegerAttr::get(IndexType::get(ctx), 0);
  unsigned writeMemrefDim = 0;
  for (unsigned baseDim : llvm::seq(baseRank)) {
    bool wasDropped = (hasSubview && droppedDims.test(baseDim));
    int64_t vectorDim = !wasDropped ? vectorDimForWriteDim[writeMemrefDim] : -1;
    int64_t extent = 1;
    if (vectorDim >= 0) {
      int64_t dimSize = vecTy.getDimSize(vectorDim);
      if (dimSize == ShapedType::kDynamic)
        return false;
      extent = dimSize;
    }
    Value writeIndex = !wasDropped ? writeIndices[writeMemrefDim] : Value();
    OpFoldResult offset =
        hasSubview ? offsets[baseDim] : OpFoldResult(zeroAttr);
    if (!loadIndexWithinWriteRange(loadIndices[baseDim], offset, writeIndex,
                                   extent, ivsMap))
      return false;
    if (!wasDropped)
      ++writeMemrefDim;
  }

  return true;
}

/// Recognize scalar memref.load of an element produced by a
/// vector.transfer_write
static bool loadMatchesVectorWrite(memref::LoadOp loadOp,
                                   vector::TransferWriteOp writeOp,
                                   const IRMapping &ivsMap) {
  auto vecTy = dyn_cast<VectorType>(writeOp.getVector().getType());
  if (!vecTy)
    return false;

  unsigned writeRank = writeOp.getIndices().size();
  AffineMap permutationMap = writeOp.getPermutationMap();
  if (!permutationMap.isProjectedPermutation() ||
      permutationMap.getNumResults() != vecTy.getRank() ||
      permutationMap.getNumDims() != writeRank)
    return false;

  SmallVector<int64_t> vectorDimForWriteDim(writeRank, -1);
  for (unsigned vecDim = 0; vecDim < permutationMap.getNumResults(); ++vecDim) {
    auto dimExpr = dyn_cast<AffineDimExpr>(permutationMap.getResult(vecDim));
    if (!dimExpr)
      return false;
    unsigned writeDim = dimExpr.getPosition();
    if (writeDim >= writeRank || vectorDimForWriteDim[writeDim] != -1)
      return false;
    vectorDimForWriteDim[writeDim] = vecDim;
  }

  return isLoadOnWrittenVector(loadOp, writeOp.getBase(), writeOp.getIndices(),
                               vecTy, vectorDimForWriteDim, ivsMap);
}

/// Recognize scalar memref.load of an element produced by a vector.store
static bool loadMatchesVectorStore(memref::LoadOp loadOp,
                                   vector::StoreOp storeOp,
                                   const IRMapping &ivsMap) {
  auto vecTy = dyn_cast<VectorType>(storeOp.getValueToStore().getType());
  if (!vecTy)
    return false;

  unsigned writeRank = storeOp.getIndices().size();
  if (vecTy.getRank() > writeRank)
    return false;

  SmallVector<int64_t> vectorDimForWriteDim(writeRank, -1);
  unsigned vecRank = vecTy.getRank();
  for (unsigned i = 0; i < vecRank; ++i) {
    unsigned writeDim = writeRank - vecRank + i;
    vectorDimForWriteDim[writeDim] = i;
  }

  return isLoadOnWrittenVector(loadOp, storeOp.getBase(), storeOp.getIndices(),
                               vecTy, vectorDimForWriteDim, ivsMap);
}

/// Check if both operations access the same positions of the same
/// buffer, but one of the two does it through a rank-reducing full subview of
/// the buffer (the other's base). EXAMPLE:
///  memref.store %a, %buf[%c0, %i, %j] : memref<1x2x2xf32>
///  %alias = memref.subview %buf[0, 0, 0][1, 2, 2][1, 1, 1]: memref<1x2x2xf32>
///                           to memref<2x2xf32>
///  %val = memref.load %alias[%i, %j] : memref<2x2xf32>
template <typename OpTy1, typename OpTy2>
static bool opsAccessSameIndicesViaRankReducingSubview(
    OpTy1 op1, OpTy2 op2, const IRMapping &firstToSecondPloopIVsMap,
    OpBuilder &b) {
  auto base1 = cast<MemrefValue>(getBaseMemref(op1));
  auto base2 = cast<MemrefValue>(getBaseMemref(op2));
  if (!base1 || !base2)
    return false;

  auto accessThroughTrivialSubviewIsSame =
      [&b](memref::SubViewOp subView, ValueRange subViewAccess,
           ValueRange sourceAccess, const IRMapping &ivsMap) -> bool {
    SmallVector<Value> resolvedSubviewAccess;
    LogicalResult resolved = resolveSourceIndicesRankReducingSubview(
        subView.getLoc(), b, subView, subViewAccess, resolvedSubviewAccess);
    if (failed(resolved) ||
        (resolvedSubviewAccess.size() != sourceAccess.size()))
      return false;
    for (auto [dimIdx, resolvedIndex] :
         llvm::enumerate(resolvedSubviewAccess)) {
      if (!matchPattern(resolvedIndex, m_Zero()) &&
          !valsAreEquivalent(resolvedIndex, sourceAccess[dimIdx], ivsMap))
        return false;
    }
    return true;
  };

  // Case 1: op1 uses a subview of op2's base.
  if (auto subView = base1.template getDefiningOp<memref::SubViewOp>();
      subView &&
      memref::isSameViewOrTrivialAlias(
          base2, cast<MemrefValue>(subView.getSource())) &&
      accessThroughTrivialSubviewIsSame(subView, op1.getIndices(),
                                        op2.getIndices(),
                                        firstToSecondPloopIVsMap))
    return true;

  // Case 2: op2 uses a subview of op1's base.
  if (auto subView = base2.template getDefiningOp<memref::SubViewOp>();
      subView &&
      memref::isSameViewOrTrivialAlias(
          base1, cast<MemrefValue>(subView.getSource())) &&
      accessThroughTrivialSubviewIsSame(subView, op2.getIndices(),
                                        op1.getIndices(),
                                        firstToSecondPloopIVsMap))
    return true;

  return false;
}

/// Check if both memory read/write operations access the same indices
/// (considering also the mapping of induction variables from the first to the
/// second parallel loop).
template <typename OpTy1, typename OpTy2>
static bool opsAccessSameIndices(OpTy1 op1, OpTy2 op2,
                                 const IRMapping &loopsIVsMap, OpBuilder &b) {
  auto indices1 = op1.getIndices();
  auto indices2 = op2.getIndices();
  if (indices1.size() != indices2.size())
    return opsAccessSameIndicesViaRankReducingSubview(op1, op2, loopsIVsMap, b);
  for (auto [idx1, idx2] : llvm::zip(indices1, indices2)) {
    if (!valsAreEquivalent(idx1, idx2, loopsIVsMap))
      return false;
  }
  return true;
}

/// Check if the loadOp reads from the same memory location (same buffer,
/// same indices and same properties) as written by the storeOp.
static bool
loadsFromSameMemoryLocationWrittenBy(Operation *loadOp, Operation *storeOp,
                                     const IRMapping &firstToSecondPloopIVsMap,
                                     OpBuilder &b) {
  if (!loadOp || !storeOp)
    return false;
  // Support only these memory-reading ops for now
  if (!isa<memref::LoadOp, vector::TransferReadOp, vector::LoadOp>(loadOp))
    return false;
  bool accessSameMemory =
      llvm::TypeSwitch<Operation *, bool>(loadOp)
          .Case([&](memref::LoadOp memLoadOp) {
            if (auto memStoreOp = dyn_cast<memref::StoreOp>(storeOp))
              return opsAccessSameIndices(memLoadOp, memStoreOp,
                                          firstToSecondPloopIVsMap, b);
            if (auto vecWriteOp = dyn_cast<vector::TransferWriteOp>(storeOp))
              return loadMatchesVectorWrite(memLoadOp, vecWriteOp,
                                            firstToSecondPloopIVsMap);
            if (auto vecStoreOp = dyn_cast<vector::StoreOp>(storeOp))
              return loadMatchesVectorStore(memLoadOp, vecStoreOp,
                                            firstToSecondPloopIVsMap);
            return false;
          })
          .Case([&](vector::TransferReadOp vecReadOp) {
            auto vecWriteOp = dyn_cast<vector::TransferWriteOp>(storeOp);
            if (!vecWriteOp)
              return false;
            return opsAccessSameIndices(vecReadOp, vecWriteOp,
                                        firstToSecondPloopIVsMap, b) &&
                   (vecReadOp.getMask() == vecWriteOp.getMask()) &&
                   (vecReadOp.getInBounds() == vecWriteOp.getInBounds());
          })
          .Case([&](vector::LoadOp vecLoadOp) {
            auto vecStoreOp = dyn_cast<vector::StoreOp>(storeOp);
            if (!vecStoreOp)
              return false;
            return opsAccessSameIndices(vecLoadOp, vecStoreOp,
                                        firstToSecondPloopIVsMap, b) &&
                   (vecLoadOp.getAlignment() == vecStoreOp.getAlignment());
          })
          .Default([](Operation *) { return false; });
  return accessSameMemory;
}

static Value getStoreOpTargetBuffer(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case([&](memref::StoreOp storeOp) { return storeOp.getMemRef(); })
      .Case([&](vector::TransferWriteOp writeOp) { return writeOp.getBase(); })
      .Case([&](vector::StoreOp vecStoreOp) { return vecStoreOp.getBase(); })
      .Default([](Operation *) { return Value(); });
}

/// To be called when `mayAlias(val1, val2)` is true. Check if the potential
/// aliasing between the loadOp and storeOp can be resolved by analyzing their
/// access patterns.
static bool canResolveAlias(Operation *loadOp, Operation *storeOp,
                            const IRMapping &loopsIVsMap) {
  if (auto transfWriteOp = dyn_cast<vector::TransferWriteOp>(storeOp);
      transfWriteOp && isa<memref::LoadOp>(loadOp))
    return loadMatchesVectorWrite(cast<memref::LoadOp>(loadOp), transfWriteOp,
                                  loopsIVsMap);
  if (auto vecStoreOp = dyn_cast<vector::StoreOp>(storeOp);
      vecStoreOp && isa<memref::LoadOp>(loadOp))
    return loadMatchesVectorStore(cast<memref::LoadOp>(loadOp), vecStoreOp,
                                  loopsIVsMap);
  return false;
}

/// Check that the parallel loops have no mixed access to the same buffers.
/// Return `true` if the second parallel loop does not read or write the buffers
/// written by the first loop using different indices.
static bool haveNoDataDependenciesExceptSameIndex(
    ParallelOp firstPloop, ParallelOp secondPloop,
    const IRMapping &firstToSecondPloopIndices,
    llvm::function_ref<bool(Value, Value)> mayAlias, OpBuilder &b) {
  // Map buffers to their store/write ops in the firstPloop
  DenseMap<Value, SmallVector<Operation *>> bufferStoresInFirstPloop;
  // Record all the memory buffers used in store/write ops found in firstPloop
  llvm::SmallSetVector<Value, 4> buffersWrittenInFirstPloop;

  auto collectStoreOpsInWalk = [&](Operation *op) {
    auto memOpInterf = dyn_cast_if_present<MemoryEffectOpInterface>(op);
    // Ignore ops that don't write to memory
    if (!memOpInterf || (!memOpInterf.hasEffect<MemoryEffects::Write>() &&
                         !memOpInterf.hasEffect<MemoryEffects::Free>()))
      return WalkResult::advance();

    // Only these memory-writing ops are supported for now:
    // memref.store, vector.transfer_write, vector.store
    Value storeOpBase = getStoreOpTargetBuffer(op);
    if (!storeOpBase)
      return WalkResult::interrupt();

    // Expect the base operand to be a Memref
    MemrefValue storeOpBaseMemref = dyn_cast<MemrefValue>(storeOpBase);
    if (!storeOpBaseMemref)
      return WalkResult::interrupt();
    // Get the original memref buffer, skipping full view-like ops
    Value buffer = memref::skipFullyAliasingOperations(storeOpBaseMemref);
    bufferStoresInFirstPloop[buffer].push_back(op);
    buffersWrittenInFirstPloop.insert(buffer);
    return WalkResult::advance();
  };

  // Walk the first parallel loop to collect all store/write ops and their
  // target buffers
  if (firstPloop.getBody()->walk(collectStoreOpsInWalk).wasInterrupted())
    return false;

  // Check that this load/read op encountered while walking the second parallel
  // loop does not have incompatible data dependencies with the store/write ops
  // collected from the first parallel loop: the loops can be fused only if in
  // the 2nd loop there are no loads/stores from/to the buffers written in the
  // 1st loop, except when on the same exact memory location (same indices) as
  // written in the 1st loop.
  auto checkLoadInWalkHasNoIncompatibleDataDeps = [&](Operation *loadOp) {
    auto memOpInterf = dyn_cast_if_present<MemoryEffectOpInterface>(loadOp);
    // To be conservative, we should stop on ops that don't advertise their
    // memory effects. However, many ops don't implement MemoryEffectOpInterface
    // yet, so for now we just skip them.
    // TODO: once more ops add MemoryEffectOpInterface, interrupt the walk here.
    if (!memOpInterf &&
        !loadOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>())
      return WalkResult::advance();
    // Ignore ops that don't read from memory, and wrapping ops that have nested
    // memory effects (e.g. loops, conditionals) as they will be analyzed when
    // visiting their nested ops.
    if ((!memOpInterf &&
         loadOp->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) ||
        (memOpInterf && !memOpInterf.hasEffect<MemoryEffects::Read>()))
      return WalkResult::advance();
    // Support only these memory-reading ops for now
    if (!isa<memref::LoadOp, vector::TransferReadOp, vector::LoadOp>(loadOp) ||
        !isa<MemrefValue>(loadOp->getOperand(0)))
      return WalkResult::interrupt();

    MemrefValue loadOpBase = cast<MemrefValue>(loadOp->getOperand(0));
    MemrefValue loadedOrigBuf = memref::skipFullyAliasingOperations(loadOpBase);

    for (Value storedMem : buffersWrittenInFirstPloop)
      if ((storedMem != loadedOrigBuf) && mayAlias(storedMem, loadedOrigBuf) &&
          !llvm::all_of(bufferStoresInFirstPloop[storedMem],
                        [&](Operation *storeOp) {
                          return canResolveAlias(loadOp, storeOp,
                                                 firstToSecondPloopIndices);
                        })) {
        return WalkResult::interrupt();
      }

    auto writeOpsIt = bufferStoresInFirstPloop.find(loadedOrigBuf);
    if (writeOpsIt == bufferStoresInFirstPloop.end())
      return WalkResult::advance();
    // Store/write ops to this buffer in the firstPloop
    SmallVector<mlir::Operation *> &writeOps = writeOpsIt->second;

    // If the first loop has no writes to this buffer, continue
    if (writeOps.empty())
      return WalkResult::advance();

    Operation *writeOp = writeOps.front();

    // In the first parallel loop, multiple writes to the same memref are
    // allowed only on the same memory location
    if (!llvm::all_of(writeOps, [&](Operation *otherWriteOp) {
          return opsWriteSameMemLocation(writeOp, otherWriteOp);
        })) {
      return WalkResult::interrupt();
    }

    // Check that the load in secondPloop reads from the same memory location as
    // written by the corresponding store in firstPloop
    if (!loadsFromSameMemoryLocationWrittenBy(loadOp, writeOp,
                                              firstToSecondPloopIndices, b)) {
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  };

  // Walk the second parallel loop to check load/read ops against the stores
  // collected from the first parallel loop.
  return !secondPloop.getBody()
              ->walk(checkLoadInWalkHasNoIncompatibleDataDeps)
              .wasInterrupted();
}

/// Check that in each loop there are no read ops on the buffers written
/// by the other loop, except when reading from the same exact memory location
/// (same indices) as written in the other loop.
static bool
noIncompatibleDataDependencies(ParallelOp firstPloop, ParallelOp secondPloop,
                               const IRMapping &firstToSecondPloopIndices,
                               llvm::function_ref<bool(Value, Value)> mayAlias,
                               OpBuilder &b) {
  if (!haveNoDataDependenciesExceptSameIndex(
          firstPloop, secondPloop, firstToSecondPloopIndices, mayAlias, b))
    return false;

  IRMapping secondToFirstPloopIndices;
  secondToFirstPloopIndices.map(secondPloop.getBody()->getArguments(),
                                firstPloop.getBody()->getArguments());
  return haveNoDataDependenciesExceptSameIndex(
      secondPloop, firstPloop, secondToFirstPloopIndices, mayAlias, b);
}

/// Check if fusion of the two parallel loops is legal:
/// i.e. no nested parallel loops, equal iteration spaces,
/// and no incompatible data dependencies between the loops.
static bool isFusionLegal(ParallelOp firstPloop, ParallelOp secondPloop,
                          const IRMapping &firstToSecondPloopIndices,
                          llvm::function_ref<bool(Value, Value)> mayAlias,
                          OpBuilder &b) {
  return !hasNestedParallelOp(firstPloop) &&
         !hasNestedParallelOp(secondPloop) &&
         equalIterationSpaces(firstPloop, secondPloop) &&
         noIncompatibleDataDependencies(firstPloop, secondPloop,
                                        firstToSecondPloopIndices, mayAlias, b);
}

/// Prepend operations of firstPloop's body into secondPloop's body.
/// Update secondPloop with new loop.
static void fuseIfLegal(ParallelOp firstPloop, ParallelOp &secondPloop,
                        OpBuilder builder,
                        llvm::function_ref<bool(Value, Value)> mayAlias) {
  Block *block1 = firstPloop.getBody();
  Block *block2 = secondPloop.getBody();
  IRMapping firstToSecondPloopIndices;
  firstToSecondPloopIndices.map(block1->getArguments(), block2->getArguments());

  if (!isFusionLegal(firstPloop, secondPloop, firstToSecondPloopIndices,
                     mayAlias, builder))
    return;

  DominanceInfo dom;
  // We are fusing first loop into second, make sure there are no users of the
  // first loop results between loops.
  for (Operation *user : firstPloop->getUsers())
    if (!dom.properlyDominates(secondPloop, user, /*enclosingOpOk*/ false))
      return;

  ValueRange inits1 = firstPloop.getInitVals();
  ValueRange inits2 = secondPloop.getInitVals();

  SmallVector<Value> newInitVars(inits1.begin(), inits1.end());
  newInitVars.append(inits2.begin(), inits2.end());

  IRRewriter b(builder);
  b.setInsertionPoint(secondPloop);
  auto newSecondPloop = ParallelOp::create(
      b, secondPloop.getLoc(), secondPloop.getLowerBound(),
      secondPloop.getUpperBound(), secondPloop.getStep(), newInitVars);

  Block *newBlock = newSecondPloop.getBody();
  auto term1 = cast<ReduceOp>(block1->getTerminator());
  auto term2 = cast<ReduceOp>(block2->getTerminator());

  b.inlineBlockBefore(block2, newBlock, newBlock->begin(),
                      newBlock->getArguments());
  b.inlineBlockBefore(block1, newBlock, newBlock->begin(),
                      newBlock->getArguments());

  ValueRange results = newSecondPloop.getResults();
  if (!results.empty()) {
    b.setInsertionPointToEnd(newBlock);

    ValueRange reduceArgs1 = term1.getOperands();
    ValueRange reduceArgs2 = term2.getOperands();
    SmallVector<Value> newReduceArgs(reduceArgs1.begin(), reduceArgs1.end());
    newReduceArgs.append(reduceArgs2.begin(), reduceArgs2.end());

    auto newReduceOp = scf::ReduceOp::create(b, term2.getLoc(), newReduceArgs);

    for (auto &&[i, reg] : llvm::enumerate(llvm::concat<Region>(
             term1.getReductions(), term2.getReductions()))) {
      Block &oldRedBlock = reg.front();
      Block &newRedBlock = newReduceOp.getReductions()[i].front();
      b.inlineBlockBefore(&oldRedBlock, &newRedBlock, newRedBlock.begin(),
                          newRedBlock.getArguments());
    }

    firstPloop.replaceAllUsesWith(results.take_front(inits1.size()));
    secondPloop.replaceAllUsesWith(results.take_back(inits2.size()));
  }
  term1->erase();
  term2->erase();
  firstPloop.erase();
  secondPloop.erase();
  secondPloop = newSecondPloop;
}

void mlir::scf::naivelyFuseParallelOps(
    Region &region, llvm::function_ref<bool(Value, Value)> mayAlias) {
  OpBuilder b(region);
  // Consider every single block and attempt to fuse adjacent loops.
  SmallVector<SmallVector<ParallelOp>, 1> ploopChains;
  for (auto &block : region) {
    ploopChains.clear();
    ploopChains.push_back({});

    // Not using `walk()` to traverse only top-level parallel loops and also
    // make sure that there are no side-effecting ops between the parallel
    // loops.
    bool noSideEffects = true;
    for (auto &op : block) {
      if (auto ploop = dyn_cast<ParallelOp>(op)) {
        if (noSideEffects) {
          ploopChains.back().push_back(ploop);
        } else {
          ploopChains.push_back({ploop});
          noSideEffects = true;
        }
        continue;
      }
      // TODO: Handle region side effects properly.
      noSideEffects &= isMemoryEffectFree(&op) && op.getNumRegions() == 0;
    }
    for (MutableArrayRef<ParallelOp> ploops : ploopChains) {
      for (int i = 0, e = ploops.size(); i + 1 < e; ++i)
        fuseIfLegal(ploops[i], ploops[i + 1], b, mayAlias);
    }
  }
}

namespace {
struct ParallelLoopFusion
    : public impl::SCFParallelLoopFusionBase<ParallelLoopFusion> {
  void runOnOperation() override {
    auto &aa = getAnalysis<AliasAnalysis>();

    auto mayAlias = [&](Value val1, Value val2) -> bool {
      // If the memref is defined in one of the parallel loops body, careful
      // alias analysis is needed.
      // TODO: check if this is still needed as a separate check.
      auto val1Def = val1.getDefiningOp();
      auto val2Def = val2.getDefiningOp();
      auto val1Loop =
          val1Def ? val1Def->getParentOfType<ParallelOp>() : nullptr;
      auto val2Loop =
          val2Def ? val2Def->getParentOfType<ParallelOp>() : nullptr;
      if (val1Loop != val2Loop)
        return true;

      return !aa.alias(val1, val2).isNo();
    };

    getOperation()->walk([&](Operation *child) {
      for (Region &region : child->getRegions())
        naivelyFuseParallelOps(region, mayAlias);
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createParallelLoopFusionPass() {
  return std::make_unique<ParallelLoopFusion>();
}
