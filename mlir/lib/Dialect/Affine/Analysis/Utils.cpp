//===- Utils.cpp ---- Misc utilities for analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous analysis routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "analysis-utils"

using namespace mlir;
using namespace presburger;

using llvm::SmallDenseMap;

/// Populates 'loops' with IVs of the loops surrounding 'op' ordered from
/// the outermost 'affine.for' operation to the innermost one.
void mlir::getLoopIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops) {
  auto *currOp = op.getParentOp();
  AffineForOp currAffineForOp;
  // Traverse up the hierarchy collecting all 'affine.for' operation while
  // skipping over 'affine.if' operations.
  while (currOp) {
    if (AffineForOp currAffineForOp = dyn_cast<AffineForOp>(currOp))
      loops->push_back(currAffineForOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(loops->begin(), loops->end());
}

/// Populates 'ops' with IVs of the loops surrounding `op`, along with
/// `affine.if` operations interleaved between these loops, ordered from the
/// outermost `affine.for` operation to the innermost one.
void mlir::getEnclosingAffineForAndIfOps(Operation &op,
                                         SmallVectorImpl<Operation *> *ops) {
  ops->clear();
  Operation *currOp = op.getParentOp();

  // Traverse up the hierarchy collecting all `affine.for` and `affine.if`
  // operations.
  while (currOp) {
    if (isa<AffineIfOp, AffineForOp>(currOp))
      ops->push_back(currOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(ops->begin(), ops->end());
}

// Populates 'cst' with FlatAffineValueConstraints which represent original
// domain of the loop bounds that define 'ivs'.
LogicalResult
ComputationSliceState::getSourceAsConstraints(FlatAffineValueConstraints &cst) {
  assert(!ivs.empty() && "Cannot have a slice without its IVs");
  cst.reset(/*numDims=*/ivs.size(), /*numSymbols=*/0, /*numLocals=*/0, ivs);
  for (Value iv : ivs) {
    AffineForOp loop = getForInductionVarOwner(iv);
    assert(loop && "Expected affine for");
    if (failed(cst.addAffineForOpDomain(loop)))
      return failure();
  }
  return success();
}

// Populates 'cst' with FlatAffineValueConstraints which represent slice bounds.
LogicalResult
ComputationSliceState::getAsConstraints(FlatAffineValueConstraints *cst) {
  assert(!lbOperands.empty());
  // Adds src 'ivs' as dimension identifiers in 'cst'.
  unsigned numDims = ivs.size();
  // Adds operands (dst ivs and symbols) as symbols in 'cst'.
  unsigned numSymbols = lbOperands[0].size();

  SmallVector<Value, 4> values(ivs);
  // Append 'ivs' then 'operands' to 'values'.
  values.append(lbOperands[0].begin(), lbOperands[0].end());
  cst->reset(numDims, numSymbols, 0, values);

  // Add loop bound constraints for values which are loop IVs of the destination
  // of fusion and equality constraints for symbols which are constants.
  for (unsigned i = numDims, end = values.size(); i < end; ++i) {
    Value value = values[i];
    assert(cst->containsId(value) && "value expected to be present");
    if (isValidSymbol(value)) {
      // Check if the symbol is a constant.
      if (auto cOp = value.getDefiningOp<arith::ConstantIndexOp>())
        cst->addBound(FlatAffineConstraints::EQ, value, cOp.value());
    } else if (auto loop = getForInductionVarOwner(value)) {
      if (failed(cst->addAffineForOpDomain(loop)))
        return failure();
    }
  }

  // Add slices bounds on 'ivs' using maps 'lbs'/'ubs' with 'lbOperands[0]'
  LogicalResult ret = cst->addSliceBounds(ivs, lbs, ubs, lbOperands[0]);
  assert(succeeded(ret) &&
         "should not fail as we never have semi-affine slice maps");
  (void)ret;
  return success();
}

// Clears state bounds and operand state.
void ComputationSliceState::clearBounds() {
  lbs.clear();
  ubs.clear();
  lbOperands.clear();
  ubOperands.clear();
}

void ComputationSliceState::dump() const {
  llvm::errs() << "\tIVs:\n";
  for (Value iv : ivs)
    llvm::errs() << "\t\t" << iv << "\n";

  llvm::errs() << "\tLBs:\n";
  for (auto &en : llvm::enumerate(lbs)) {
    llvm::errs() << "\t\t" << en.value() << "\n";
    llvm::errs() << "\t\tOperands:\n";
    for (Value lbOp : lbOperands[en.index()])
      llvm::errs() << "\t\t\t" << lbOp << "\n";
  }

  llvm::errs() << "\tUBs:\n";
  for (auto &en : llvm::enumerate(ubs)) {
    llvm::errs() << "\t\t" << en.value() << "\n";
    llvm::errs() << "\t\tOperands:\n";
    for (Value ubOp : ubOperands[en.index()])
      llvm::errs() << "\t\t\t" << ubOp << "\n";
  }
}

/// Fast check to determine if the computation slice is maximal. Returns true if
/// each slice dimension maps to an existing dst dimension and both the src
/// and the dst loops for those dimensions have the same bounds. Returns false
/// if both the src and the dst loops don't have the same bounds. Returns
/// llvm::None if none of the above can be proven.
Optional<bool> ComputationSliceState::isSliceMaximalFastCheck() const {
  assert(lbs.size() == ubs.size() && !lbs.empty() && !ivs.empty() &&
         "Unexpected number of lbs, ubs and ivs in slice");

  for (unsigned i = 0, end = lbs.size(); i < end; ++i) {
    AffineMap lbMap = lbs[i];
    AffineMap ubMap = ubs[i];

    // Check if this slice is just an equality along this dimension.
    if (!lbMap || !ubMap || lbMap.getNumResults() != 1 ||
        ubMap.getNumResults() != 1 ||
        lbMap.getResult(0) + 1 != ubMap.getResult(0) ||
        // The condition above will be true for maps describing a single
        // iteration (e.g., lbMap.getResult(0) = 0, ubMap.getResult(0) = 1).
        // Make sure we skip those cases by checking that the lb result is not
        // just a constant.
        lbMap.getResult(0).isa<AffineConstantExpr>())
      return llvm::None;

    // Limited support: we expect the lb result to be just a loop dimension for
    // now.
    AffineDimExpr result = lbMap.getResult(0).dyn_cast<AffineDimExpr>();
    if (!result)
      return llvm::None;

    // Retrieve dst loop bounds.
    AffineForOp dstLoop =
        getForInductionVarOwner(lbOperands[i][result.getPosition()]);
    if (!dstLoop)
      return llvm::None;
    AffineMap dstLbMap = dstLoop.getLowerBoundMap();
    AffineMap dstUbMap = dstLoop.getUpperBoundMap();

    // Retrieve src loop bounds.
    AffineForOp srcLoop = getForInductionVarOwner(ivs[i]);
    assert(srcLoop && "Expected affine for");
    AffineMap srcLbMap = srcLoop.getLowerBoundMap();
    AffineMap srcUbMap = srcLoop.getUpperBoundMap();

    // Limited support: we expect simple src and dst loops with a single
    // constant component per bound for now.
    if (srcLbMap.getNumResults() != 1 || srcUbMap.getNumResults() != 1 ||
        dstLbMap.getNumResults() != 1 || dstUbMap.getNumResults() != 1)
      return llvm::None;

    AffineExpr srcLbResult = srcLbMap.getResult(0);
    AffineExpr dstLbResult = dstLbMap.getResult(0);
    AffineExpr srcUbResult = srcUbMap.getResult(0);
    AffineExpr dstUbResult = dstUbMap.getResult(0);
    if (!srcLbResult.isa<AffineConstantExpr>() ||
        !srcUbResult.isa<AffineConstantExpr>() ||
        !dstLbResult.isa<AffineConstantExpr>() ||
        !dstUbResult.isa<AffineConstantExpr>())
      return llvm::None;

    // Check if src and dst loop bounds are the same. If not, we can guarantee
    // that the slice is not maximal.
    if (srcLbResult != dstLbResult || srcUbResult != dstUbResult ||
        srcLoop.getStep() != dstLoop.getStep())
      return false;
  }

  return true;
}

/// Returns true if it is deterministically verified that the original iteration
/// space of the slice is contained within the new iteration space that is
/// created after fusing 'this' slice into its destination.
Optional<bool> ComputationSliceState::isSliceValid() {
  // Fast check to determine if the slice is valid. If the following conditions
  // are verified to be true, slice is declared valid by the fast check:
  // 1. Each slice loop is a single iteration loop bound in terms of a single
  //    destination loop IV.
  // 2. Loop bounds of the destination loop IV (from above) and those of the
  //    source loop IV are exactly the same.
  // If the fast check is inconclusive or false, we proceed with a more
  // expensive analysis.
  // TODO: Store the result of the fast check, as it might be used again in
  // `canRemoveSrcNodeAfterFusion`.
  Optional<bool> isValidFastCheck = isSliceMaximalFastCheck();
  if (isValidFastCheck.hasValue() && isValidFastCheck.getValue())
    return true;

  // Create constraints for the source loop nest using which slice is computed.
  FlatAffineValueConstraints srcConstraints;
  // TODO: Store the source's domain to avoid computation at each depth.
  if (failed(getSourceAsConstraints(srcConstraints))) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute source's domain\n");
    return llvm::None;
  }
  // As the set difference utility currently cannot handle symbols in its
  // operands, validity of the slice cannot be determined.
  if (srcConstraints.getNumSymbolIds() > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot handle symbols in source domain\n");
    return llvm::None;
  }
  // TODO: Handle local ids in the source domains while using the 'projectOut'
  // utility below. Currently, aligning is not done assuming that there will be
  // no local ids in the source domain.
  if (srcConstraints.getNumLocalIds() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot handle locals in source domain\n");
    return llvm::None;
  }

  // Create constraints for the slice loop nest that would be created if the
  // fusion succeeds.
  FlatAffineValueConstraints sliceConstraints;
  if (failed(getAsConstraints(&sliceConstraints))) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute slice's domain\n");
    return llvm::None;
  }

  // Projecting out every dimension other than the 'ivs' to express slice's
  // domain completely in terms of source's IVs.
  sliceConstraints.projectOut(ivs.size(),
                              sliceConstraints.getNumIds() - ivs.size());

  LLVM_DEBUG(llvm::dbgs() << "Domain of the source of the slice:\n");
  LLVM_DEBUG(srcConstraints.dump());
  LLVM_DEBUG(llvm::dbgs() << "Domain of the slice if this fusion succeeds "
                             "(expressed in terms of its source's IVs):\n");
  LLVM_DEBUG(sliceConstraints.dump());

  // TODO: Store 'srcSet' to avoid recalculating for each depth.
  PresburgerSet srcSet(srcConstraints);
  PresburgerSet sliceSet(sliceConstraints);
  PresburgerSet diffSet = sliceSet.subtract(srcSet);

  if (!diffSet.isIntegerEmpty()) {
    LLVM_DEBUG(llvm::dbgs() << "Incorrect slice\n");
    return false;
  }
  return true;
}

/// Returns true if the computation slice encloses all the iterations of the
/// sliced loop nest. Returns false if it does not. Returns llvm::None if it
/// cannot determine if the slice is maximal or not.
Optional<bool> ComputationSliceState::isMaximal() const {
  // Fast check to determine if the computation slice is maximal. If the result
  // is inconclusive, we proceed with a more expensive analysis.
  Optional<bool> isMaximalFastCheck = isSliceMaximalFastCheck();
  if (isMaximalFastCheck.hasValue())
    return isMaximalFastCheck;

  // Create constraints for the src loop nest being sliced.
  FlatAffineValueConstraints srcConstraints;
  srcConstraints.reset(/*numDims=*/ivs.size(), /*numSymbols=*/0,
                       /*numLocals=*/0, ivs);
  for (Value iv : ivs) {
    AffineForOp loop = getForInductionVarOwner(iv);
    assert(loop && "Expected affine for");
    if (failed(srcConstraints.addAffineForOpDomain(loop)))
      return llvm::None;
  }

  // Create constraints for the slice using the dst loop nest information. We
  // retrieve existing dst loops from the lbOperands.
  SmallVector<Value, 8> consumerIVs;
  for (Value lbOp : lbOperands[0])
    if (getForInductionVarOwner(lbOp))
      consumerIVs.push_back(lbOp);

  // Add empty IV Values for those new loops that are not equalities and,
  // therefore, are not yet materialized in the IR.
  for (int i = consumerIVs.size(), end = ivs.size(); i < end; ++i)
    consumerIVs.push_back(Value());

  FlatAffineValueConstraints sliceConstraints;
  sliceConstraints.reset(/*numDims=*/consumerIVs.size(), /*numSymbols=*/0,
                         /*numLocals=*/0, consumerIVs);

  if (failed(sliceConstraints.addDomainFromSliceMaps(lbs, ubs, lbOperands[0])))
    return llvm::None;

  if (srcConstraints.getNumDimIds() != sliceConstraints.getNumDimIds())
    // Constraint dims are different. The integer set difference can't be
    // computed so we don't know if the slice is maximal.
    return llvm::None;

  // Compute the difference between the src loop nest and the slice integer
  // sets.
  PresburgerSet srcSet(srcConstraints);
  PresburgerSet sliceSet(sliceConstraints);
  PresburgerSet diffSet = srcSet.subtract(sliceSet);
  return diffSet.isIntegerEmpty();
}

unsigned MemRefRegion::getRank() const {
  return memref.getType().cast<MemRefType>().getRank();
}

Optional<int64_t> MemRefRegion::getConstantBoundingSizeAndShape(
    SmallVectorImpl<int64_t> *shape, std::vector<SmallVector<int64_t, 4>> *lbs,
    SmallVectorImpl<int64_t> *lbDivisors) const {
  auto memRefType = memref.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  if (shape)
    shape->reserve(rank);

  assert(rank == cst.getNumDimIds() && "inconsistent memref region");

  // Use a copy of the region constraints that has upper/lower bounds for each
  // memref dimension with static size added to guard against potential
  // over-approximation from projection or union bounding box. We may not add
  // this on the region itself since they might just be redundant constraints
  // that will need non-trivials means to eliminate.
  FlatAffineConstraints cstWithShapeBounds(cst);
  for (unsigned r = 0; r < rank; r++) {
    cstWithShapeBounds.addBound(FlatAffineConstraints::LB, r, 0);
    int64_t dimSize = memRefType.getDimSize(r);
    if (ShapedType::isDynamic(dimSize))
      continue;
    cstWithShapeBounds.addBound(FlatAffineConstraints::UB, r, dimSize - 1);
  }

  // Find a constant upper bound on the extent of this memref region along each
  // dimension.
  int64_t numElements = 1;
  int64_t diffConstant;
  int64_t lbDivisor;
  for (unsigned d = 0; d < rank; d++) {
    SmallVector<int64_t, 4> lb;
    Optional<int64_t> diff =
        cstWithShapeBounds.getConstantBoundOnDimSize(d, &lb, &lbDivisor);
    if (diff.hasValue()) {
      diffConstant = diff.getValue();
      assert(diffConstant >= 0 && "Dim size bound can't be negative");
      assert(lbDivisor > 0);
    } else {
      // If no constant bound is found, then it can always be bound by the
      // memref's dim size if the latter has a constant size along this dim.
      auto dimSize = memRefType.getDimSize(d);
      if (dimSize == -1)
        return None;
      diffConstant = dimSize;
      // Lower bound becomes 0.
      lb.resize(cstWithShapeBounds.getNumSymbolIds() + 1, 0);
      lbDivisor = 1;
    }
    numElements *= diffConstant;
    if (lbs) {
      lbs->push_back(lb);
      assert(lbDivisors && "both lbs and lbDivisor or none");
      lbDivisors->push_back(lbDivisor);
    }
    if (shape) {
      shape->push_back(diffConstant);
    }
  }
  return numElements;
}

void MemRefRegion::getLowerAndUpperBound(unsigned pos, AffineMap &lbMap,
                                         AffineMap &ubMap) const {
  assert(pos < cst.getNumDimIds() && "invalid position");
  auto memRefType = memref.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();

  assert(rank == cst.getNumDimIds() && "inconsistent memref region");

  auto boundPairs = cst.getLowerAndUpperBound(
      pos, /*offset=*/0, /*num=*/rank, cst.getNumDimAndSymbolIds(),
      /*localExprs=*/{}, memRefType.getContext());
  lbMap = boundPairs.first;
  ubMap = boundPairs.second;
  assert(lbMap && "lower bound for a region must exist");
  assert(ubMap && "upper bound for a region must exist");
  assert(lbMap.getNumInputs() == cst.getNumDimAndSymbolIds() - rank);
  assert(ubMap.getNumInputs() == cst.getNumDimAndSymbolIds() - rank);
}

LogicalResult MemRefRegion::unionBoundingBox(const MemRefRegion &other) {
  assert(memref == other.memref);
  return cst.unionBoundingBox(*other.getConstraints());
}

/// Computes the memory region accessed by this memref with the region
/// represented as constraints symbolic/parametric in 'loopDepth' loops
/// surrounding opInst and any additional Function symbols.
//  For example, the memref region for this load operation at loopDepth = 1 will
//  be as below:
//
//    affine.for %i = 0 to 32 {
//      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        load %A[%ii]
//      }
//    }
//
// region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineConstraints symbolic in %i.
//
// TODO: extend this to any other memref dereferencing ops
// (dma_start, dma_wait).
LogicalResult MemRefRegion::compute(Operation *op, unsigned loopDepth,
                                    const ComputationSliceState *sliceState,
                                    bool addMemRefDimBounds) {
  assert((isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) &&
         "affine read/write op expected");

  MemRefAccess access(op);
  memref = access.memref;
  write = access.isStore();

  unsigned rank = access.getRank();

  LLVM_DEBUG(llvm::dbgs() << "MemRefRegion::compute: " << *op
                          << "depth: " << loopDepth << "\n";);

  // 0-d memrefs.
  if (rank == 0) {
    SmallVector<AffineForOp, 4> ivs;
    getLoopIVs(*op, &ivs);
    assert(loopDepth <= ivs.size() && "invalid 'loopDepth'");
    // The first 'loopDepth' IVs are symbols for this region.
    ivs.resize(loopDepth);
    SmallVector<Value, 4> regionSymbols;
    extractForInductionVars(ivs, &regionSymbols);
    // A 0-d memref has a 0-d region.
    cst.reset(rank, loopDepth, /*numLocals=*/0, regionSymbols);
    return success();
  }

  // Build the constraints for this region.
  AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  AffineMap accessMap = accessValueMap.getAffineMap();

  unsigned numDims = accessMap.getNumDims();
  unsigned numSymbols = accessMap.getNumSymbols();
  unsigned numOperands = accessValueMap.getNumOperands();
  // Merge operands with slice operands.
  SmallVector<Value, 4> operands;
  operands.resize(numOperands);
  for (unsigned i = 0; i < numOperands; ++i)
    operands[i] = accessValueMap.getOperand(i);

  if (sliceState != nullptr) {
    operands.reserve(operands.size() + sliceState->lbOperands[0].size());
    // Append slice operands to 'operands' as symbols.
    for (auto extraOperand : sliceState->lbOperands[0]) {
      if (!llvm::is_contained(operands, extraOperand)) {
        operands.push_back(extraOperand);
        numSymbols++;
      }
    }
  }
  // We'll first associate the dims and symbols of the access map to the dims
  // and symbols resp. of cst. This will change below once cst is
  // fully constructed out.
  cst.reset(numDims, numSymbols, 0, operands);

  // Add equality constraints.
  // Add inequalities for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    auto operand = operands[i];
    if (auto loop = getForInductionVarOwner(operand)) {
      // Note that cst can now have more dimensions than accessMap if the
      // bounds expressions involve outer loops or other symbols.
      // TODO: rewrite this to use getInstIndexSet; this way
      // conditionals will be handled when the latter supports it.
      if (failed(cst.addAffineForOpDomain(loop)))
        return failure();
    } else {
      // Has to be a valid symbol.
      auto symbol = operand;
      assert(isValidSymbol(symbol));
      // Check if the symbol is a constant.
      if (auto *op = symbol.getDefiningOp()) {
        if (auto constOp = dyn_cast<arith::ConstantIndexOp>(op)) {
          cst.addBound(FlatAffineConstraints::EQ, symbol, constOp.value());
        }
      }
    }
  }

  // Add lower/upper bounds on loop IVs using bounds from 'sliceState'.
  if (sliceState != nullptr) {
    // Add dim and symbol slice operands.
    for (auto operand : sliceState->lbOperands[0]) {
      cst.addInductionVarOrTerminalSymbol(operand);
    }
    // Add upper/lower bounds from 'sliceState' to 'cst'.
    LogicalResult ret =
        cst.addSliceBounds(sliceState->ivs, sliceState->lbs, sliceState->ubs,
                           sliceState->lbOperands[0]);
    assert(succeeded(ret) &&
           "should not fail as we never have semi-affine slice maps");
    (void)ret;
  }

  // Add access function equalities to connect loop IVs to data dimensions.
  if (failed(cst.composeMap(&accessValueMap))) {
    op->emitError("getMemRefRegion: compose affine map failed");
    LLVM_DEBUG(accessValueMap.getAffineMap().dump());
    return failure();
  }

  // Set all identifiers appearing after the first 'rank' identifiers as
  // symbolic identifiers - so that the ones corresponding to the memref
  // dimensions are the dimensional identifiers for the memref region.
  cst.setDimSymbolSeparation(cst.getNumDimAndSymbolIds() - rank);

  // Eliminate any loop IVs other than the outermost 'loopDepth' IVs, on which
  // this memref region is symbolic.
  SmallVector<AffineForOp, 4> enclosingIVs;
  getLoopIVs(*op, &enclosingIVs);
  assert(loopDepth <= enclosingIVs.size() && "invalid loop depth");
  enclosingIVs.resize(loopDepth);
  SmallVector<Value, 4> ids;
  cst.getValues(cst.getNumDimIds(), cst.getNumDimAndSymbolIds(), &ids);
  for (auto id : ids) {
    AffineForOp iv;
    if ((iv = getForInductionVarOwner(id)) &&
        !llvm::is_contained(enclosingIVs, iv)) {
      cst.projectOut(id);
    }
  }

  // Project out any local variables (these would have been added for any
  // mod/divs).
  cst.projectOut(cst.getNumDimAndSymbolIds(), cst.getNumLocalIds());

  // Constant fold any symbolic identifiers.
  cst.constantFoldIdRange(/*pos=*/cst.getNumDimIds(),
                          /*num=*/cst.getNumSymbolIds());

  assert(cst.getNumDimIds() == rank && "unexpected MemRefRegion format");

  // Add upper/lower bounds for each memref dimension with static size
  // to guard against potential over-approximation from projection.
  // TODO: Support dynamic memref dimensions.
  if (addMemRefDimBounds) {
    auto memRefType = memref.getType().cast<MemRefType>();
    for (unsigned r = 0; r < rank; r++) {
      cst.addBound(FlatAffineConstraints::LB, /*pos=*/r, /*value=*/0);
      if (memRefType.isDynamicDim(r))
        continue;
      cst.addBound(FlatAffineConstraints::UB, /*pos=*/r,
                   memRefType.getDimSize(r) - 1);
    }
  }
  cst.removeTrivialRedundancy();

  LLVM_DEBUG(llvm::dbgs() << "Memory region:\n");
  LLVM_DEBUG(cst.dump());
  return success();
}

static unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Returns the size of the region.
Optional<int64_t> MemRefRegion::getRegionSize() {
  auto memRefType = memref.getType().cast<MemRefType>();

  if (!memRefType.getLayout().isIdentity()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-identity layout map not yet supported\n");
    return false;
  }

  // Indices to use for the DmaStart op.
  // Indices for the original memref being DMAed from/to.
  SmallVector<Value, 4> memIndices;
  // Indices for the faster buffer being DMAed into/from.
  SmallVector<Value, 4> bufIndices;

  // Compute the extents of the buffer.
  Optional<int64_t> numElements = getConstantBoundingSizeAndShape();
  if (!numElements.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Dynamic shapes not yet supported\n");
    return None;
  }
  return getMemRefEltSizeInBytes(memRefType) * numElements.getValue();
}

/// Returns the size of memref data in bytes if it's statically shaped, None
/// otherwise.  If the element of the memref has vector type, takes into account
/// size of the vector as well.
//  TODO: improve/complete this when we have target data.
Optional<uint64_t> mlir::getMemRefSizeInBytes(MemRefType memRefType) {
  if (!memRefType.hasStaticShape())
    return None;
  auto elementType = memRefType.getElementType();
  if (!elementType.isIntOrFloat() && !elementType.isa<VectorType>())
    return None;

  uint64_t sizeInBytes = getMemRefEltSizeInBytes(memRefType);
  for (unsigned i = 0, e = memRefType.getRank(); i < e; i++) {
    sizeInBytes = sizeInBytes * memRefType.getDimSize(i);
  }
  return sizeInBytes;
}

template <typename LoadOrStoreOp>
LogicalResult mlir::boundCheckLoadOrStoreOp(LoadOrStoreOp loadOrStoreOp,
                                            bool emitError) {
  static_assert(llvm::is_one_of<LoadOrStoreOp, AffineReadOpInterface,
                                AffineWriteOpInterface>::value,
                "argument should be either a AffineReadOpInterface or a "
                "AffineWriteOpInterface");

  Operation *op = loadOrStoreOp.getOperation();
  MemRefRegion region(op->getLoc());
  if (failed(region.compute(op, /*loopDepth=*/0, /*sliceState=*/nullptr,
                            /*addMemRefDimBounds=*/false)))
    return success();

  LLVM_DEBUG(llvm::dbgs() << "Memory region");
  LLVM_DEBUG(region.getConstraints()->dump());

  bool outOfBounds = false;
  unsigned rank = loadOrStoreOp.getMemRefType().getRank();

  // For each dimension, check for out of bounds.
  for (unsigned r = 0; r < rank; r++) {
    FlatAffineConstraints ucst(*region.getConstraints());

    // Intersect memory region with constraint capturing out of bounds (both out
    // of upper and out of lower), and check if the constraint system is
    // feasible. If it is, there is at least one point out of bounds.
    SmallVector<int64_t, 4> ineq(rank + 1, 0);
    int64_t dimSize = loadOrStoreOp.getMemRefType().getDimSize(r);
    // TODO: handle dynamic dim sizes.
    if (dimSize == -1)
      continue;

    // Check for overflow: d_i >= memref dim size.
    ucst.addBound(FlatAffineConstraints::LB, r, dimSize);
    outOfBounds = !ucst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp.emitOpError()
          << "memref out of upper bound access along dimension #" << (r + 1);
    }

    // Check for a negative index.
    FlatAffineConstraints lcst(*region.getConstraints());
    std::fill(ineq.begin(), ineq.end(), 0);
    // d_i <= -1;
    lcst.addBound(FlatAffineConstraints::UB, r, -1);
    outOfBounds = !lcst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp.emitOpError()
          << "memref out of lower bound access along dimension #" << (r + 1);
    }
  }
  return failure(outOfBounds);
}

// Explicitly instantiate the template so that the compiler knows we need them!
template LogicalResult
mlir::boundCheckLoadOrStoreOp(AffineReadOpInterface loadOp, bool emitError);
template LogicalResult
mlir::boundCheckLoadOrStoreOp(AffineWriteOpInterface storeOp, bool emitError);

// Returns in 'positions' the Block positions of 'op' in each ancestor
// Block from the Block containing operation, stopping at 'limitBlock'.
static void findInstPosition(Operation *op, Block *limitBlock,
                             SmallVectorImpl<unsigned> *positions) {
  Block *block = op->getBlock();
  while (block != limitBlock) {
    // FIXME: This algorithm is unnecessarily O(n) and should be improved to not
    // rely on linear scans.
    int instPosInBlock = std::distance(block->begin(), op->getIterator());
    positions->push_back(instPosInBlock);
    op = block->getParentOp();
    block = op->getBlock();
  }
  std::reverse(positions->begin(), positions->end());
}

// Returns the Operation in a possibly nested set of Blocks, where the
// position of the operation is represented by 'positions', which has a
// Block position for each level of nesting.
static Operation *getInstAtPosition(ArrayRef<unsigned> positions,
                                    unsigned level, Block *block) {
  unsigned i = 0;
  for (auto &op : *block) {
    if (i != positions[level]) {
      ++i;
      continue;
    }
    if (level == positions.size() - 1)
      return &op;
    if (auto childAffineForOp = dyn_cast<AffineForOp>(op))
      return getInstAtPosition(positions, level + 1,
                               childAffineForOp.getBody());

    for (auto &region : op.getRegions()) {
      for (auto &b : region)
        if (auto *ret = getInstAtPosition(positions, level + 1, &b))
          return ret;
    }
    return nullptr;
  }
  return nullptr;
}

// Adds loop IV bounds to 'cst' for loop IVs not found in 'ivs'.
static LogicalResult addMissingLoopIVBounds(SmallPtrSet<Value, 8> &ivs,
                                            FlatAffineValueConstraints *cst) {
  for (unsigned i = 0, e = cst->getNumDimIds(); i < e; ++i) {
    auto value = cst->getValue(i);
    if (ivs.count(value) == 0) {
      assert(isForInductionVar(value));
      auto loop = getForInductionVarOwner(value);
      if (failed(cst->addAffineForOpDomain(loop)))
        return failure();
    }
  }
  return success();
}

/// Returns the innermost common loop depth for the set of operations in 'ops'.
// TODO: Move this to LoopUtils.
unsigned mlir::getInnermostCommonLoopDepth(
    ArrayRef<Operation *> ops, SmallVectorImpl<AffineForOp> *surroundingLoops) {
  unsigned numOps = ops.size();
  assert(numOps > 0 && "Expected at least one operation");

  std::vector<SmallVector<AffineForOp, 4>> loops(numOps);
  unsigned loopDepthLimit = std::numeric_limits<unsigned>::max();
  for (unsigned i = 0; i < numOps; ++i) {
    getLoopIVs(*ops[i], &loops[i]);
    loopDepthLimit =
        std::min(loopDepthLimit, static_cast<unsigned>(loops[i].size()));
  }

  unsigned loopDepth = 0;
  for (unsigned d = 0; d < loopDepthLimit; ++d) {
    unsigned i;
    for (i = 1; i < numOps; ++i) {
      if (loops[i - 1][d] != loops[i][d])
        return loopDepth;
    }
    if (surroundingLoops)
      surroundingLoops->push_back(loops[i - 1][d]);
    ++loopDepth;
  }
  return loopDepth;
}

/// Computes in 'sliceUnion' the union of all slice bounds computed at
/// 'loopDepth' between all dependent pairs of ops in 'opsA' and 'opsB', and
/// then verifies if it is valid. Returns 'SliceComputationResult::Success' if
/// union was computed correctly, an appropriate failure otherwise.
SliceComputationResult
mlir::computeSliceUnion(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                        unsigned loopDepth, unsigned numCommonLoops,
                        bool isBackwardSlice,
                        ComputationSliceState *sliceUnion) {
  // Compute the union of slice bounds between all pairs in 'opsA' and
  // 'opsB' in 'sliceUnionCst'.
  FlatAffineValueConstraints sliceUnionCst;
  assert(sliceUnionCst.getNumDimAndSymbolIds() == 0);
  std::vector<std::pair<Operation *, Operation *>> dependentOpPairs;
  for (auto *i : opsA) {
    MemRefAccess srcAccess(i);
    for (auto *j : opsB) {
      MemRefAccess dstAccess(j);
      if (srcAccess.memref != dstAccess.memref)
        continue;
      // Check if 'loopDepth' exceeds nesting depth of src/dst ops.
      if ((!isBackwardSlice && loopDepth > getNestingDepth(i)) ||
          (isBackwardSlice && loopDepth > getNestingDepth(j))) {
        LLVM_DEBUG(llvm::dbgs() << "Invalid loop depth\n");
        return SliceComputationResult::GenericFailure;
      }

      bool readReadAccesses = isa<AffineReadOpInterface>(srcAccess.opInst) &&
                              isa<AffineReadOpInterface>(dstAccess.opInst);
      FlatAffineValueConstraints dependenceConstraints;
      // Check dependence between 'srcAccess' and 'dstAccess'.
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, dstAccess, /*loopDepth=*/numCommonLoops + 1,
          &dependenceConstraints, /*dependenceComponents=*/nullptr,
          /*allowRAR=*/readReadAccesses);
      if (result.value == DependenceResult::Failure) {
        LLVM_DEBUG(llvm::dbgs() << "Dependence check failed\n");
        return SliceComputationResult::GenericFailure;
      }
      if (result.value == DependenceResult::NoDependence)
        continue;
      dependentOpPairs.emplace_back(i, j);

      // Compute slice bounds for 'srcAccess' and 'dstAccess'.
      ComputationSliceState tmpSliceState;
      mlir::getComputationSliceState(i, j, &dependenceConstraints, loopDepth,
                                     isBackwardSlice, &tmpSliceState);

      if (sliceUnionCst.getNumDimAndSymbolIds() == 0) {
        // Initialize 'sliceUnionCst' with the bounds computed in previous step.
        if (failed(tmpSliceState.getAsConstraints(&sliceUnionCst))) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Unable to compute slice bound constraints\n");
          return SliceComputationResult::GenericFailure;
        }
        assert(sliceUnionCst.getNumDimAndSymbolIds() > 0);
        continue;
      }

      // Compute constraints for 'tmpSliceState' in 'tmpSliceCst'.
      FlatAffineValueConstraints tmpSliceCst;
      if (failed(tmpSliceState.getAsConstraints(&tmpSliceCst))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to compute slice bound constraints\n");
        return SliceComputationResult::GenericFailure;
      }

      // Align coordinate spaces of 'sliceUnionCst' and 'tmpSliceCst' if needed.
      if (!sliceUnionCst.areIdsAlignedWithOther(tmpSliceCst)) {

        // Pre-constraint id alignment: record loop IVs used in each constraint
        // system.
        SmallPtrSet<Value, 8> sliceUnionIVs;
        for (unsigned k = 0, l = sliceUnionCst.getNumDimIds(); k < l; ++k)
          sliceUnionIVs.insert(sliceUnionCst.getValue(k));
        SmallPtrSet<Value, 8> tmpSliceIVs;
        for (unsigned k = 0, l = tmpSliceCst.getNumDimIds(); k < l; ++k)
          tmpSliceIVs.insert(tmpSliceCst.getValue(k));

        sliceUnionCst.mergeAndAlignIdsWithOther(/*offset=*/0, &tmpSliceCst);

        // Post-constraint id alignment: add loop IV bounds missing after
        // id alignment to constraint systems. This can occur if one constraint
        // system uses an loop IV that is not used by the other. The call
        // to unionBoundingBox below expects constraints for each Loop IV, even
        // if they are the unsliced full loop bounds added here.
        if (failed(addMissingLoopIVBounds(sliceUnionIVs, &sliceUnionCst)))
          return SliceComputationResult::GenericFailure;
        if (failed(addMissingLoopIVBounds(tmpSliceIVs, &tmpSliceCst)))
          return SliceComputationResult::GenericFailure;
      }
      // Compute union bounding box of 'sliceUnionCst' and 'tmpSliceCst'.
      if (sliceUnionCst.getNumLocalIds() > 0 ||
          tmpSliceCst.getNumLocalIds() > 0 ||
          failed(sliceUnionCst.unionBoundingBox(tmpSliceCst))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to compute union bounding box of slice bounds\n");
        return SliceComputationResult::GenericFailure;
      }
    }
  }

  // Empty union.
  if (sliceUnionCst.getNumDimAndSymbolIds() == 0)
    return SliceComputationResult::GenericFailure;

  // Gather loops surrounding ops from loop nest where slice will be inserted.
  SmallVector<Operation *, 4> ops;
  for (auto &dep : dependentOpPairs) {
    ops.push_back(isBackwardSlice ? dep.second : dep.first);
  }
  SmallVector<AffineForOp, 4> surroundingLoops;
  unsigned innermostCommonLoopDepth =
      getInnermostCommonLoopDepth(ops, &surroundingLoops);
  if (loopDepth > innermostCommonLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "Exceeds max loop depth\n");
    return SliceComputationResult::GenericFailure;
  }

  // Store 'numSliceLoopIVs' before converting dst loop IVs to dims.
  unsigned numSliceLoopIVs = sliceUnionCst.getNumDimIds();

  // Convert any dst loop IVs which are symbol identifiers to dim identifiers.
  sliceUnionCst.convertLoopIVSymbolsToDims();
  sliceUnion->clearBounds();
  sliceUnion->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceUnion->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get slice bounds from slice union constraints 'sliceUnionCst'.
  sliceUnionCst.getSliceBounds(/*offset=*/0, numSliceLoopIVs,
                               opsA[0]->getContext(), &sliceUnion->lbs,
                               &sliceUnion->ubs);

  // Add slice bound operands of union.
  SmallVector<Value, 4> sliceBoundOperands;
  sliceUnionCst.getValues(numSliceLoopIVs,
                          sliceUnionCst.getNumDimAndSymbolIds(),
                          &sliceBoundOperands);

  // Copy src loop IVs from 'sliceUnionCst' to 'sliceUnion'.
  sliceUnion->ivs.clear();
  sliceUnionCst.getValues(0, numSliceLoopIVs, &sliceUnion->ivs);

  // Set loop nest insertion point to block start at 'loopDepth'.
  sliceUnion->insertPoint =
      isBackwardSlice
          ? surroundingLoops[loopDepth - 1].getBody()->begin()
          : std::prev(surroundingLoops[loopDepth - 1].getBody()->end());

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceUnion->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceUnion->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Check if the slice computed is valid. Return success only if it is verified
  // that the slice is valid, otherwise return appropriate failure status.
  Optional<bool> isSliceValid = sliceUnion->isSliceValid();
  if (!isSliceValid.hasValue()) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot determine if the slice is valid\n");
    return SliceComputationResult::GenericFailure;
  }
  if (!isSliceValid.getValue())
    return SliceComputationResult::IncorrectSliceFailure;

  return SliceComputationResult::Success;
}

// TODO: extend this to handle multiple result maps.
static Optional<uint64_t> getConstDifference(AffineMap lbMap, AffineMap ubMap) {
  assert(lbMap.getNumResults() == 1 && "expected single result bound map");
  assert(ubMap.getNumResults() == 1 && "expected single result bound map");
  assert(lbMap.getNumDims() == ubMap.getNumDims());
  assert(lbMap.getNumSymbols() == ubMap.getNumSymbols());
  AffineExpr lbExpr(lbMap.getResult(0));
  AffineExpr ubExpr(ubMap.getResult(0));
  auto loopSpanExpr = simplifyAffineExpr(ubExpr - lbExpr, lbMap.getNumDims(),
                                         lbMap.getNumSymbols());
  auto cExpr = loopSpanExpr.dyn_cast<AffineConstantExpr>();
  if (!cExpr)
    return None;
  return cExpr.getValue();
}

// Builds a map 'tripCountMap' from AffineForOp to constant trip count for loop
// nest surrounding represented by slice loop bounds in 'slice'. Returns true
// on success, false otherwise (if a non-constant trip count was encountered).
// TODO: Make this work with non-unit step loops.
bool mlir::buildSliceTripCountMap(
    const ComputationSliceState &slice,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountMap) {
  unsigned numSrcLoopIVs = slice.ivs.size();
  // Populate map from AffineForOp -> trip count
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    AffineForOp forOp = getForInductionVarOwner(slice.ivs[i]);
    auto *op = forOp.getOperation();
    AffineMap lbMap = slice.lbs[i];
    AffineMap ubMap = slice.ubs[i];
    // If lower or upper bound maps are null or provide no results, it implies
    // that source loop was not at all sliced, and the entire loop will be a
    // part of the slice.
    if (!lbMap || lbMap.getNumResults() == 0 || !ubMap ||
        ubMap.getNumResults() == 0) {
      // The iteration of src loop IV 'i' was not sliced. Use full loop bounds.
      if (forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound()) {
        (*tripCountMap)[op] =
            forOp.getConstantUpperBound() - forOp.getConstantLowerBound();
        continue;
      }
      Optional<uint64_t> maybeConstTripCount = getConstantTripCount(forOp);
      if (maybeConstTripCount.hasValue()) {
        (*tripCountMap)[op] = maybeConstTripCount.getValue();
        continue;
      }
      return false;
    }
    Optional<uint64_t> tripCount = getConstDifference(lbMap, ubMap);
    // Slice bounds are created with a constant ub - lb difference.
    if (!tripCount.hasValue())
      return false;
    (*tripCountMap)[op] = tripCount.getValue();
  }
  return true;
}

// Return the number of iterations in the given slice.
uint64_t mlir::getSliceIterationCount(
    const llvm::SmallDenseMap<Operation *, uint64_t, 8> &sliceTripCountMap) {
  uint64_t iterCount = 1;
  for (const auto &count : sliceTripCountMap) {
    iterCount *= count.second;
  }
  return iterCount;
}

const char *const kSliceFusionBarrierAttrName = "slice_fusion_barrier";
// Computes slice bounds by projecting out any loop IVs from
// 'dependenceConstraints' at depth greater than 'loopDepth', and computes slice
// bounds in 'sliceState' which represent the one loop nest's IVs in terms of
// the other loop nest's IVs, symbols and constants (using 'isBackwardsSlice').
void mlir::getComputationSliceState(
    Operation *depSourceOp, Operation *depSinkOp,
    FlatAffineValueConstraints *dependenceConstraints, unsigned loopDepth,
    bool isBackwardSlice, ComputationSliceState *sliceState) {
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*depSourceOp, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getLoopIVs(*depSinkOp, &dstLoopIVs);
  unsigned numDstLoopIVs = dstLoopIVs.size();

  assert((!isBackwardSlice && loopDepth <= numSrcLoopIVs) ||
         (isBackwardSlice && loopDepth <= numDstLoopIVs));

  // Project out dimensions other than those up to 'loopDepth'.
  unsigned pos = isBackwardSlice ? numSrcLoopIVs + loopDepth : loopDepth;
  unsigned num =
      isBackwardSlice ? numDstLoopIVs - loopDepth : numSrcLoopIVs - loopDepth;
  dependenceConstraints->projectOut(pos, num);

  // Add slice loop IV values to 'sliceState'.
  unsigned offset = isBackwardSlice ? 0 : loopDepth;
  unsigned numSliceLoopIVs = isBackwardSlice ? numSrcLoopIVs : numDstLoopIVs;
  dependenceConstraints->getValues(offset, offset + numSliceLoopIVs,
                                   &sliceState->ivs);

  // Set up lower/upper bound affine maps for the slice.
  sliceState->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceState->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get bounds for slice IVs in terms of other IVs, symbols, and constants.
  dependenceConstraints->getSliceBounds(offset, numSliceLoopIVs,
                                        depSourceOp->getContext(),
                                        &sliceState->lbs, &sliceState->ubs);

  // Set up bound operands for the slice's lower and upper bounds.
  SmallVector<Value, 4> sliceBoundOperands;
  unsigned numDimsAndSymbols = dependenceConstraints->getNumDimAndSymbolIds();
  for (unsigned i = 0; i < numDimsAndSymbols; ++i) {
    if (i < offset || i >= offset + numSliceLoopIVs) {
      sliceBoundOperands.push_back(dependenceConstraints->getValue(i));
    }
  }

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceState->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceState->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Set destination loop nest insertion point to block start at 'dstLoopDepth'.
  sliceState->insertPoint =
      isBackwardSlice ? dstLoopIVs[loopDepth - 1].getBody()->begin()
                      : std::prev(srcLoopIVs[loopDepth - 1].getBody()->end());

  llvm::SmallDenseSet<Value, 8> sequentialLoops;
  if (isa<AffineReadOpInterface>(depSourceOp) &&
      isa<AffineReadOpInterface>(depSinkOp)) {
    // For read-read access pairs, clear any slice bounds on sequential loops.
    // Get sequential loops in loop nest rooted at 'srcLoopIVs[0]'.
    getSequentialLoops(isBackwardSlice ? srcLoopIVs[0] : dstLoopIVs[0],
                       &sequentialLoops);
  }
  auto getSliceLoop = [&](unsigned i) {
    return isBackwardSlice ? srcLoopIVs[i] : dstLoopIVs[i];
  };
  auto isInnermostInsertion = [&]() {
    return (isBackwardSlice ? loopDepth >= srcLoopIVs.size()
                            : loopDepth >= dstLoopIVs.size());
  };
  llvm::SmallDenseMap<Operation *, uint64_t, 8> sliceTripCountMap;
  auto srcIsUnitSlice = [&]() {
    return (buildSliceTripCountMap(*sliceState, &sliceTripCountMap) &&
            (getSliceIterationCount(sliceTripCountMap) == 1));
  };
  // Clear all sliced loop bounds beginning at the first sequential loop, or
  // first loop with a slice fusion barrier attribute..

  for (unsigned i = 0; i < numSliceLoopIVs; ++i) {
    Value iv = getSliceLoop(i).getInductionVar();
    if (sequentialLoops.count(iv) == 0 &&
        getSliceLoop(i)->getAttr(kSliceFusionBarrierAttrName) == nullptr)
      continue;
    // Skip reset of bounds of reduction loop inserted in the destination loop
    // that meets the following conditions:
    //    1. Slice is  single trip count.
    //    2. Loop bounds of the source and destination match.
    //    3. Is being inserted at the innermost insertion point.
    Optional<bool> isMaximal = sliceState->isMaximal();
    if (isLoopParallelAndContainsReduction(getSliceLoop(i)) &&
        isInnermostInsertion() && srcIsUnitSlice() && isMaximal.hasValue() &&
        isMaximal.getValue())
      continue;
    for (unsigned j = i; j < numSliceLoopIVs; ++j) {
      sliceState->lbs[j] = AffineMap();
      sliceState->ubs[j] = AffineMap();
    }
    break;
  }
}

/// Creates a computation slice of the loop nest surrounding 'srcOpInst',
/// updates the slice loop bounds with any non-null bound maps specified in
/// 'sliceState', and inserts this slice into the loop nest surrounding
/// 'dstOpInst' at loop depth 'dstLoopDepth'.
// TODO: extend the slicing utility to compute slices that
// aren't necessarily a one-to-one relation b/w the source and destination. The
// relation between the source and destination could be many-to-many in general.
// TODO: the slice computation is incorrect in the cases
// where the dependence from the source to the destination does not cover the
// entire destination index set. Subtract out the dependent destination
// iterations from destination index set and check for emptiness --- this is one
// solution.
AffineForOp
mlir::insertBackwardComputationSlice(Operation *srcOpInst, Operation *dstOpInst,
                                     unsigned dstLoopDepth,
                                     ComputationSliceState *sliceState) {
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getLoopIVs(*dstOpInst, &dstLoopIVs);
  unsigned dstLoopIVsSize = dstLoopIVs.size();
  if (dstLoopDepth > dstLoopIVsSize) {
    dstOpInst->emitError("invalid destination loop depth");
    return AffineForOp();
  }

  // Find the op block positions of 'srcOpInst' within 'srcLoopIVs'.
  SmallVector<unsigned, 4> positions;
  // TODO: This code is incorrect since srcLoopIVs can be 0-d.
  findInstPosition(srcOpInst, srcLoopIVs[0]->getBlock(), &positions);

  // Clone src loop nest and insert it a the beginning of the operation block
  // of the loop at 'dstLoopDepth' in 'dstLoopIVs'.
  auto dstAffineForOp = dstLoopIVs[dstLoopDepth - 1];
  OpBuilder b(dstAffineForOp.getBody(), dstAffineForOp.getBody()->begin());
  auto sliceLoopNest =
      cast<AffineForOp>(b.clone(*srcLoopIVs[0].getOperation()));

  Operation *sliceInst =
      getInstAtPosition(positions, /*level=*/0, sliceLoopNest.getBody());
  // Get loop nest surrounding 'sliceInst'.
  SmallVector<AffineForOp, 4> sliceSurroundingLoops;
  getLoopIVs(*sliceInst, &sliceSurroundingLoops);

  // Sanity check.
  unsigned sliceSurroundingLoopsSize = sliceSurroundingLoops.size();
  (void)sliceSurroundingLoopsSize;
  assert(dstLoopDepth + numSrcLoopIVs >= sliceSurroundingLoopsSize);
  unsigned sliceLoopLimit = dstLoopDepth + numSrcLoopIVs;
  (void)sliceLoopLimit;
  assert(sliceLoopLimit >= sliceSurroundingLoopsSize);

  // Update loop bounds for loops in 'sliceLoopNest'.
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    auto forOp = sliceSurroundingLoops[dstLoopDepth + i];
    if (AffineMap lbMap = sliceState->lbs[i])
      forOp.setLowerBound(sliceState->lbOperands[i], lbMap);
    if (AffineMap ubMap = sliceState->ubs[i])
      forOp.setUpperBound(sliceState->ubOperands[i], ubMap);
  }
  return sliceLoopNest;
}

// Constructs  MemRefAccess populating it with the memref, its indices and
// opinst from 'loadOrStoreOpInst'.
MemRefAccess::MemRefAccess(Operation *loadOrStoreOpInst) {
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    auto loadMemrefType = loadOp.getMemRefType();
    indices.reserve(loadMemrefType.getRank());
    for (auto index : loadOp.getMapOperands()) {
      indices.push_back(index);
    }
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    auto storeMemrefType = storeOp.getMemRefType();
    indices.reserve(storeMemrefType.getRank());
    for (auto index : storeOp.getMapOperands()) {
      indices.push_back(index);
    }
  }
}

unsigned MemRefAccess::getRank() const {
  return memref.getType().cast<MemRefType>().getRank();
}

bool MemRefAccess::isStore() const {
  return isa<AffineWriteOpInterface>(opInst);
}

/// Returns the nesting depth of this statement, i.e., the number of loops
/// surrounding this statement.
unsigned mlir::getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<AffineForOp>(currOp))
      depth++;
  }
  return depth;
}

/// Equal if both affine accesses are provably equivalent (at compile
/// time) when considering the memref, the affine maps and their respective
/// operands. The equality of access functions + operands is checked by
/// subtracting fully composed value maps, and then simplifying the difference
/// using the expression flattener.
/// TODO: this does not account for aliasing of memrefs.
bool MemRefAccess::operator==(const MemRefAccess &rhs) const {
  if (memref != rhs.memref)
    return false;

  AffineValueMap diff, thisMap, rhsMap;
  getAccessMap(&thisMap);
  rhs.getAccessMap(&rhsMap);
  AffineValueMap::difference(thisMap, rhsMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned mlir::getNumCommonSurroundingLoops(Operation &a, Operation &b) {
  SmallVector<AffineForOp, 4> loopsA, loopsB;
  getLoopIVs(a, &loopsA);
  getLoopIVs(b, &loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i].getOperation() != loopsB[i].getOperation())
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

static Optional<int64_t> getMemoryFootprintBytes(Block &block,
                                                 Block::iterator start,
                                                 Block::iterator end,
                                                 int memorySpace) {
  SmallDenseMap<Value, std::unique_ptr<MemRefRegion>, 4> regions;

  // Walk this 'affine.for' operation to gather all memory regions.
  auto result = block.walk(start, end, [&](Operation *opInst) -> WalkResult {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(opInst)) {
      // Neither load nor a store op.
      return WalkResult::advance();
    }

    // Compute the memref region symbolic in any IVs enclosing this block.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(
            region->compute(opInst,
                            /*loopDepth=*/getNestingDepth(&*block.begin())))) {
      return opInst->emitError("error obtaining memory region\n");
    }

    auto it = regions.find(region->memref);
    if (it == regions.end()) {
      regions[region->memref] = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      return opInst->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region");
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return None;

  int64_t totalSizeInBytes = 0;
  for (const auto &region : regions) {
    Optional<int64_t> size = region.second->getRegionSize();
    if (!size.hasValue())
      return None;
    totalSizeInBytes += size.getValue();
  }
  return totalSizeInBytes;
}

Optional<int64_t> mlir::getMemoryFootprintBytes(AffineForOp forOp,
                                                int memorySpace) {
  auto *forInst = forOp.getOperation();
  return ::getMemoryFootprintBytes(
      *forInst->getBlock(), Block::iterator(forInst),
      std::next(Block::iterator(forInst)), memorySpace);
}

/// Returns whether a loop is parallel and contains a reduction loop.
bool mlir::isLoopParallelAndContainsReduction(AffineForOp forOp) {
  SmallVector<LoopReduction> reductions;
  if (!isLoopParallel(forOp, &reductions))
    return false;
  return !reductions.empty();
}

/// Returns in 'sequentialLoops' all sequential loops in loop nest rooted
/// at 'forOp'.
void mlir::getSequentialLoops(AffineForOp forOp,
                              llvm::SmallDenseSet<Value, 8> *sequentialLoops) {
  forOp->walk([&](Operation *op) {
    if (auto innerFor = dyn_cast<AffineForOp>(op))
      if (!isLoopParallel(innerFor))
        sequentialLoops->insert(innerFor.getInductionVar());
  });
}

IntegerSet mlir::simplifyIntegerSet(IntegerSet set) {
  FlatAffineConstraints fac(set);
  if (fac.isEmpty())
    return IntegerSet::getEmptySet(set.getNumDims(), set.getNumSymbols(),
                                   set.getContext());
  fac.removeTrivialRedundancy();

  auto simplifiedSet = fac.getAsIntegerSet(set.getContext());
  assert(simplifiedSet && "guaranteed to succeed while roundtripping");
  return simplifiedSet;
}
