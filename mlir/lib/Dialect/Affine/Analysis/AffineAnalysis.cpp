//===- AffineAnalysis.cpp - Affine structures analysis routines -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous analysis routines for affine structures
// (expressions, maps, sets), and other utilities relying on such analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "affine-analysis"

using namespace mlir;

/// Get the value that is being reduced by `pos`-th reduction in the loop if
/// such a reduction can be performed by affine parallel loops. This assumes
/// floating-point operations are commutative. On success, `kind` will be the
/// reduction kind suitable for use in affine parallel loop builder. If the
/// reduction is not supported, returns null.
static Value getSupportedReduction(AffineForOp forOp, unsigned pos,
                                   arith::AtomicRMWKind &kind) {
  SmallVector<Operation *> combinerOps;
  Value reducedVal =
      matchReduction(forOp.getRegionIterArgs(), pos, combinerOps);
  if (!reducedVal)
    return nullptr;

  // Expected only one combiner operation.
  if (combinerOps.size() > 1)
    return nullptr;

  Operation *combinerOp = combinerOps.back();
  Optional<arith::AtomicRMWKind> maybeKind =
      TypeSwitch<Operation *, Optional<arith::AtomicRMWKind>>(combinerOp)
          .Case([](arith::AddFOp) { return arith::AtomicRMWKind::addf; })
          .Case([](arith::MulFOp) { return arith::AtomicRMWKind::mulf; })
          .Case([](arith::AddIOp) { return arith::AtomicRMWKind::addi; })
          .Case([](arith::AndIOp) { return arith::AtomicRMWKind::andi; })
          .Case([](arith::OrIOp) { return arith::AtomicRMWKind::ori; })
          .Case([](arith::MulIOp) { return arith::AtomicRMWKind::muli; })
          .Case([](arith::MinFOp) { return arith::AtomicRMWKind::minf; })
          .Case([](arith::MaxFOp) { return arith::AtomicRMWKind::maxf; })
          .Case([](arith::MinSIOp) { return arith::AtomicRMWKind::mins; })
          .Case([](arith::MaxSIOp) { return arith::AtomicRMWKind::maxs; })
          .Case([](arith::MinUIOp) { return arith::AtomicRMWKind::minu; })
          .Case([](arith::MaxUIOp) { return arith::AtomicRMWKind::maxu; })
          .Default([](Operation *) -> Optional<arith::AtomicRMWKind> {
            // TODO: AtomicRMW supports other kinds of reductions this is
            // currently not detecting, add those when the need arises.
            return llvm::None;
          });
  if (!maybeKind)
    return nullptr;

  kind = *maybeKind;
  return reducedVal;
}

/// Populate `supportedReductions` with descriptors of the supported reductions.
void mlir::getSupportedReductions(
    AffineForOp forOp, SmallVectorImpl<LoopReduction> &supportedReductions) {
  unsigned numIterArgs = forOp.getNumIterOperands();
  if (numIterArgs == 0)
    return;
  supportedReductions.reserve(numIterArgs);
  for (unsigned i = 0; i < numIterArgs; ++i) {
    arith::AtomicRMWKind kind;
    if (Value value = getSupportedReduction(forOp, i, kind))
      supportedReductions.emplace_back(LoopReduction{kind, i, value});
  }
}

/// Returns true if `forOp' is a parallel loop. If `parallelReductions` is
/// provided, populates it with descriptors of the parallelizable reductions and
/// treats them as not preventing parallelization.
bool mlir::isLoopParallel(AffineForOp forOp,
                          SmallVectorImpl<LoopReduction> *parallelReductions) {
  unsigned numIterArgs = forOp.getNumIterOperands();

  // Loop is not parallel if it has SSA loop-carried dependences and reduction
  // detection is not requested.
  if (numIterArgs > 0 && !parallelReductions)
    return false;

  // Find supported reductions of requested.
  if (parallelReductions) {
    getSupportedReductions(forOp, *parallelReductions);
    // Return later to allow for identifying all parallel reductions even if the
    // loop is not parallel.
    if (parallelReductions->size() != numIterArgs)
      return false;
  }

  // Check memory dependences.
  return isLoopMemoryParallel(forOp);
}

/// Returns true if `forOp' doesn't have memory dependences preventing
/// parallelization. This function doesn't check iter_args and should be used
/// only as a building block for full parallel-checking functions.
bool mlir::isLoopMemoryParallel(AffineForOp forOp) {
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadAndStoreOps;
  auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
    else if (!isa<AffineForOp, AffineYieldOp, AffineIfOp>(op) &&
             !MemoryEffectOpInterface::hasNoEffect(op))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  // Stop early if the loop has unknown ops with side effects.
  if (walkResult.wasInterrupted())
    return false;

  // Dep check depth would be number of enclosing loops + 1.
  unsigned depth = getNestingDepth(forOp) + 1;

  // Check dependences between all pairs of ops in 'loadAndStoreOps'.
  for (auto *srcOp : loadAndStoreOps) {
    MemRefAccess srcAccess(srcOp);
    for (auto *dstOp : loadAndStoreOps) {
      MemRefAccess dstAccess(dstOp);
      FlatAffineValueConstraints dependenceConstraints;
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, dstAccess, depth, &dependenceConstraints,
          /*dependenceComponents=*/nullptr);
      if (result.value != DependenceResult::NoDependence)
        return false;
    }
  }
  return true;
}

/// Returns the sequence of AffineApplyOp Operations operation in
/// 'affineApplyOps', which are reachable via a search starting from 'operands',
/// and ending at operands which are not defined by AffineApplyOps.
// TODO: Add a method to AffineApplyOp which forward substitutes the
// AffineApplyOp into any user AffineApplyOps.
void mlir::getReachableAffineApplyOps(
    ArrayRef<Value> operands, SmallVectorImpl<Operation *> &affineApplyOps) {
  struct State {
    // The ssa value for this node in the DFS traversal.
    Value value;
    // The operand index of 'value' to explore next during DFS traversal.
    unsigned operandIndex;
  };
  SmallVector<State, 4> worklist;
  for (auto operand : operands) {
    worklist.push_back({operand, 0});
  }

  while (!worklist.empty()) {
    State &state = worklist.back();
    auto *opInst = state.value.getDefiningOp();
    // Note: getDefiningOp will return nullptr if the operand is not an
    // Operation (i.e. block argument), which is a terminator for the search.
    if (!isa_and_nonnull<AffineApplyOp>(opInst)) {
      worklist.pop_back();
      continue;
    }

    if (state.operandIndex == 0) {
      // Pre-Visit: Add 'opInst' to reachable sequence.
      affineApplyOps.push_back(opInst);
    }
    if (state.operandIndex < opInst->getNumOperands()) {
      // Visit: Add next 'affineApplyOp' operand to worklist.
      // Get next operand to visit at 'operandIndex'.
      auto nextOperand = opInst->getOperand(state.operandIndex);
      // Increment 'operandIndex' in 'state'.
      ++state.operandIndex;
      // Add 'nextOperand' to worklist.
      worklist.push_back({nextOperand, 0});
    } else {
      // Post-visit: done visiting operands AffineApplyOp, pop off stack.
      worklist.pop_back();
    }
  }
}

// Builds a system of constraints with dimensional identifiers corresponding to
// the loop IVs of the forOps appearing in that order. Any symbols founds in
// the bound operands are added as symbols in the system. Returns failure for
// the yet unimplemented cases.
// TODO: Handle non-unit steps through local variables or stride information in
// FlatAffineValueConstraints. (For eg., by using iv - lb % step = 0 and/or by
// introducing a method in FlatAffineValueConstraints
// setExprStride(ArrayRef<int64_t> expr, int64_t stride)
LogicalResult mlir::getIndexSet(MutableArrayRef<Operation *> ops,
                                FlatAffineValueConstraints *domain) {
  SmallVector<Value, 4> indices;
  SmallVector<AffineForOp, 8> forOps;

  for (Operation *op : ops) {
    assert((isa<AffineForOp, AffineIfOp>(op)) &&
           "ops should have either AffineForOp or AffineIfOp");
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op))
      forOps.push_back(forOp);
  }
  extractForInductionVars(forOps, &indices);
  // Reset while associated Values in 'indices' to the domain.
  domain->reset(forOps.size(), /*numSymbols=*/0, /*numLocals=*/0, indices);
  for (Operation *op : ops) {
    // Add constraints from forOp's bounds.
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (failed(domain->addAffineForOpDomain(forOp)))
        return failure();
    } else if (AffineIfOp ifOp = dyn_cast<AffineIfOp>(op)) {
      domain->addAffineIfOpDomain(ifOp);
    }
  }
  return success();
}

/// Computes the iteration domain for 'op' and populates 'indexSet', which
/// encapsulates the constraints involving loops surrounding 'op' and
/// potentially involving any Function symbols. The dimensional identifiers in
/// 'indexSet' correspond to the loops surrounding 'op' from outermost to
/// innermost.
static LogicalResult getOpIndexSet(Operation *op,
                                   FlatAffineValueConstraints *indexSet) {
  SmallVector<Operation *, 4> ops;
  getEnclosingAffineForAndIfOps(*op, &ops);
  return getIndexSet(ops, indexSet);
}

// Returns the number of outer loop common to 'src/dstDomain'.
// Loops common to 'src/dst' domains are added to 'commonLoops' if non-null.
static unsigned
getNumCommonLoops(const FlatAffineValueConstraints &srcDomain,
                  const FlatAffineValueConstraints &dstDomain,
                  SmallVectorImpl<AffineForOp> *commonLoops = nullptr) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned minNumLoops =
      std::min(srcDomain.getNumDimIds(), dstDomain.getNumDimIds());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (!isForInductionVar(srcDomain.getValue(i)) ||
        !isForInductionVar(dstDomain.getValue(i)) ||
        srcDomain.getValue(i) != dstDomain.getValue(i))
      break;
    if (commonLoops != nullptr)
      commonLoops->push_back(getForInductionVarOwner(srcDomain.getValue(i)));
    ++numCommonLoops;
  }
  if (commonLoops != nullptr)
    assert(commonLoops->size() == numCommonLoops);
  return numCommonLoops;
}

/// Returns Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
static Block *getCommonBlock(const MemRefAccess &srcAccess,
                             const MemRefAccess &dstAccess,
                             const FlatAffineValueConstraints &srcDomain,
                             unsigned numCommonLoops) {
  // Get the chain of ancestor blocks to the given `MemRefAccess` instance. The
  // search terminates when either an op with the `AffineScope` trait or
  // `endBlock` is reached.
  auto getChainOfAncestorBlocks = [&](const MemRefAccess &access,
                                      SmallVector<Block *, 4> &ancestorBlocks,
                                      Block *endBlock = nullptr) {
    Block *currBlock = access.opInst->getBlock();
    // Loop terminates when the currBlock is nullptr or equals to the endBlock,
    // or its parent operation holds an affine scope.
    while (currBlock && currBlock != endBlock &&
           !currBlock->getParentOp()->hasTrait<OpTrait::AffineScope>()) {
      ancestorBlocks.push_back(currBlock);
      currBlock = currBlock->getParentOp()->getBlock();
    }
  };

  if (numCommonLoops == 0) {
    Block *block = srcAccess.opInst->getBlock();
    while (!llvm::isa<FuncOp>(block->getParentOp())) {
      block = block->getParentOp()->getBlock();
    }
    return block;
  }
  Value commonForIV = srcDomain.getValue(numCommonLoops - 1);
  AffineForOp forOp = getForInductionVarOwner(commonForIV);
  assert(forOp && "commonForValue was not an induction variable");

  // Find the closest common block including those in AffineIf.
  SmallVector<Block *, 4> srcAncestorBlocks, dstAncestorBlocks;
  getChainOfAncestorBlocks(srcAccess, srcAncestorBlocks, forOp.getBody());
  getChainOfAncestorBlocks(dstAccess, dstAncestorBlocks, forOp.getBody());

  Block *commonBlock = forOp.getBody();
  for (int i = srcAncestorBlocks.size() - 1, j = dstAncestorBlocks.size() - 1;
       i >= 0 && j >= 0 && srcAncestorBlocks[i] == dstAncestorBlocks[j];
       i--, j--)
    commonBlock = srcAncestorBlocks[i];

  return commonBlock;
}

// Returns true if the ancestor operation of 'srcAccess' appears before the
// ancestor operation of 'dstAccess' in the common ancestral block. Returns
// false otherwise.
// Note that because 'srcAccess' or 'dstAccess' may be nested in conditionals,
// the function is named 'srcAppearsBeforeDstInCommonBlock'. Note that
// 'numCommonLoops' is the number of contiguous surrounding outer loops.
static bool srcAppearsBeforeDstInAncestralBlock(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    const FlatAffineValueConstraints &srcDomain, unsigned numCommonLoops) {
  // Get Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
  auto *commonBlock =
      getCommonBlock(srcAccess, dstAccess, srcDomain, numCommonLoops);
  // Check the dominance relationship between the respective ancestors of the
  // src and dst in the Block of the innermost among the common loops.
  auto *srcInst = commonBlock->findAncestorOpInBlock(*srcAccess.opInst);
  assert(srcInst != nullptr);
  auto *dstInst = commonBlock->findAncestorOpInBlock(*dstAccess.opInst);
  assert(dstInst != nullptr);

  // Determine whether dstInst comes after srcInst.
  return srcInst->isBeforeInBlock(dstInst);
}

// Adds ordering constraints to 'dependenceDomain' based on number of loops
// common to 'src/dstDomain' and requested 'loopDepth'.
// Note that 'loopDepth' cannot exceed the number of common loops plus one.
// EX: Given a loop nest of depth 2 with IVs 'i' and 'j':
// *) If 'loopDepth == 1' then one constraint is added: i' >= i + 1
// *) If 'loopDepth == 2' then two constraints are added: i == i' and j' > j + 1
// *) If 'loopDepth == 3' then two constraints are added: i == i' and j == j'
static void
addOrderingConstraints(const FlatAffineValueConstraints &srcDomain,
                       const FlatAffineValueConstraints &dstDomain,
                       unsigned loopDepth,
                       FlatAffineValueConstraints *dependenceDomain) {
  unsigned numCols = dependenceDomain->getNumCols();
  SmallVector<int64_t, 4> eq(numCols);
  unsigned numSrcDims = srcDomain.getNumDimIds();
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  unsigned numCommonLoopConstraints = std::min(numCommonLoops, loopDepth);
  for (unsigned i = 0; i < numCommonLoopConstraints; ++i) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[i] = -1;
    eq[i + numSrcDims] = 1;
    if (i == loopDepth - 1) {
      eq[numCols - 1] = -1;
      dependenceDomain->addInequality(eq);
    } else {
      dependenceDomain->addEquality(eq);
    }
  }
}

// Computes distance and direction vectors in 'dependences', by adding
// variables to 'dependenceDomain' which represent the difference of the IVs,
// eliminating all other variables, and reading off distance vectors from
// equality constraints (if possible), and direction vectors from inequalities.
static void computeDirectionVector(
    const FlatAffineValueConstraints &srcDomain,
    const FlatAffineValueConstraints &dstDomain, unsigned loopDepth,
    FlatAffineValueConstraints *dependenceDomain,
    SmallVector<DependenceComponent, 2> *dependenceComponents) {
  // Find the number of common loops shared by src and dst accesses.
  SmallVector<AffineForOp, 4> commonLoops;
  unsigned numCommonLoops =
      getNumCommonLoops(srcDomain, dstDomain, &commonLoops);
  if (numCommonLoops == 0)
    return;
  // Compute direction vectors for requested loop depth.
  unsigned numIdsToEliminate = dependenceDomain->getNumIds();
  // Add new variables to 'dependenceDomain' to represent the direction
  // constraints for each shared loop.
  dependenceDomain->insertDimId(/*pos=*/0, /*num=*/numCommonLoops);

  // Add equality constraints for each common loop, setting newly introduced
  // variable at column 'j' to the 'dst' IV minus the 'src IV.
  SmallVector<int64_t, 4> eq;
  eq.resize(dependenceDomain->getNumCols());
  unsigned numSrcDims = srcDomain.getNumDimIds();
  // Constraint variables format:
  // [num-common-loops][num-src-dim-ids][num-dst-dim-ids][num-symbols][constant]
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[j] = 1;
    eq[j + numCommonLoops] = 1;
    eq[j + numCommonLoops + numSrcDims] = -1;
    dependenceDomain->addEquality(eq);
  }

  // Eliminate all variables other than the direction variables just added.
  dependenceDomain->projectOut(numCommonLoops, numIdsToEliminate);

  // Scan each common loop variable column and set direction vectors based
  // on eliminated constraint system.
  dependenceComponents->resize(numCommonLoops);
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    (*dependenceComponents)[j].op = commonLoops[j].getOperation();
    auto lbConst =
        dependenceDomain->getConstantBound(FlatAffineConstraints::LB, j);
    (*dependenceComponents)[j].lb =
        lbConst.getValueOr(std::numeric_limits<int64_t>::min());
    auto ubConst =
        dependenceDomain->getConstantBound(FlatAffineConstraints::UB, j);
    (*dependenceComponents)[j].ub =
        ubConst.getValueOr(std::numeric_limits<int64_t>::max());
  }
}

LogicalResult MemRefAccess::getAccessRelation(FlatAffineRelation &rel) const {
  // Create set corresponding to domain of access.
  FlatAffineValueConstraints domain;
  if (failed(getOpIndexSet(opInst, &domain)))
    return failure();

  // Get access relation from access map.
  AffineValueMap accessValueMap;
  getAccessMap(&accessValueMap);
  if (failed(getRelationFromMap(accessValueMap, rel)))
    return failure();

  FlatAffineRelation domainRel(rel.getNumDomainDims(), /*numRangeDims=*/0,
                               domain);

  // Merge and align domain ids of `ret` and ids of `domain`. Since the domain
  // of the access map is a subset of the domain of access, the domain ids of
  // `ret` are guranteed to be a subset of ids of `domain`.
  for (unsigned i = 0, e = domain.getNumDimIds(); i < e; ++i) {
    unsigned loc;
    if (rel.findId(domain.getValue(i), &loc)) {
      rel.swapId(i, loc);
    } else {
      rel.insertDomainId(i);
      rel.setValue(i, domain.getValue(i));
    }
  }

  // Append domain constraints to `rel`.
  domainRel.appendRangeId(rel.getNumRangeDims());
  domainRel.mergeSymbolIds(rel);
  domainRel.mergeLocalIds(rel);
  rel.append(domainRel);

  return success();
}

// Populates 'accessMap' with composition of AffineApplyOps reachable from
// indices of MemRefAccess.
void MemRefAccess::getAccessMap(AffineValueMap *accessMap) const {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst))
    map = loadOp.getAffineMap();
  else
    map = cast<AffineWriteOpInterface>(opInst).getAffineMap();

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  accessMap->reset(map, operands);
}

// Builds a flat affine constraint system to check if there exists a dependence
// between memref accesses 'srcAccess' and 'dstAccess'.
// Returns 'NoDependence' if the accesses can be definitively shown not to
// access the same element.
// Returns 'HasDependence' if the accesses do access the same element.
// Returns 'Failure' if an error or unsupported case was encountered.
// If a dependence exists, returns in 'dependenceComponents' a direction
// vector for the dependence, with a component for each loop IV in loops
// common to both accesses (see Dependence in AffineAnalysis.h for details).
//
// The memref access dependence check is comprised of the following steps:
// *) Build access relation for each access. An access relation maps elements
//    of an iteration domain to the element(s) of an array domain accessed by
//    that iteration of the associated statement through some array reference.
// *) Compute the dependence relation by composing access relation of
//    `srcAccess` with the inverse of access relation of `dstAccess`.
//    Doing this builds a relation between iteration domain of `srcAccess`
//    to the iteration domain of `dstAccess` which access the same memory
//    location.
// *) Add ordering constraints for `srcAccess` to be accessed before
//    `dstAccess`.
//
// This method builds a constraint system with the following column format:
//
//  [src-dim-identifiers, dst-dim-identifiers, symbols, constant]
//
// For example, given the following MLIR code with "source" and "destination"
// accesses to the same memref label, and symbols %M, %N, %K:
//
//   affine.for %i0 = 0 to 100 {
//     affine.for %i1 = 0 to 50 {
//       %a0 = affine.apply
//         (d0, d1) -> (d0 * 2 - d1 * 4 + s1, d1 * 3 - s0) (%i0, %i1)[%M, %N]
//       // Source memref access.
//       store %v0, %m[%a0#0, %a0#1] : memref<4x4xf32>
//     }
//   }
//
//   affine.for %i2 = 0 to 100 {
//     affine.for %i3 = 0 to 50 {
//       %a1 = affine.apply
//         (d0, d1) -> (d0 * 7 + d1 * 9 - s1, d1 * 11 + s0) (%i2, %i3)[%K, %M]
//       // Destination memref access.
//       %v1 = load %m[%a1#0, %a1#1] : memref<4x4xf32>
//     }
//   }
//
// The access relation for `srcAccess` would be the following:
//
//   [src_dim0, src_dim1, mem_dim0, mem_dim1,  %N,   %M,  const]
//       2        -4       -1         0         1     0     0     = 0
//       0         3        0        -1         0    -1     0     = 0
//       1         0        0         0         0     0     0    >= 0
//      -1         0        0         0         0     0     100  >= 0
//       0         1        0         0         0     0     0    >= 0
//       0        -1        0         0         0     0     50   >= 0
//
//  The access relation for `dstAccess` would be the following:
//
//   [dst_dim0, dst_dim1, mem_dim0, mem_dim1,  %M,   %K,  const]
//       7         9       -1         0        -1     0     0     = 0
//       0         11       0        -1         0    -1     0     = 0
//       1         0        0         0         0     0     0    >= 0
//      -1         0        0         0         0     0     100  >= 0
//       0         1        0         0         0     0     0    >= 0
//       0        -1        0         0         0     0     50   >= 0
//
//  The equalities in the above relations correspond to the access maps while
//  the inequalities corresspond to the iteration domain constraints.
//
// The dependence relation formed:
//
//   [src_dim0, src_dim1, dst_dim0, dst_dim1,  %M,   %N,   %K,  const]
//      2         -4        -7        -9        1     1     0     0    = 0
//      0          3         0        -11      -1     0     1     0    = 0
//       1         0         0         0        0     0     0     0    >= 0
//      -1         0         0         0        0     0     0     100  >= 0
//       0         1         0         0        0     0     0     0    >= 0
//       0        -1         0         0        0     0     0     50   >= 0
//       0         0         1         0        0     0     0     0    >= 0
//       0         0        -1         0        0     0     0     100  >= 0
//       0         0         0         1        0     0     0     0    >= 0
//       0         0         0        -1        0     0     0     50   >= 0
//
//
// TODO: Support AffineExprs mod/floordiv/ceildiv.
DependenceResult mlir::checkMemrefAccessDependence(
    const MemRefAccess &srcAccess, const MemRefAccess &dstAccess,
    unsigned loopDepth, FlatAffineValueConstraints *dependenceConstraints,
    SmallVector<DependenceComponent, 2> *dependenceComponents, bool allowRAR) {
  LLVM_DEBUG(llvm::dbgs() << "Checking for dependence at depth: "
                          << Twine(loopDepth) << " between:\n";);
  LLVM_DEBUG(srcAccess.opInst->dump(););
  LLVM_DEBUG(dstAccess.opInst->dump(););

  // Return 'NoDependence' if these accesses do not access the same memref.
  if (srcAccess.memref != dstAccess.memref)
    return DependenceResult::NoDependence;

  // Return 'NoDependence' if one of these accesses is not an
  // AffineWriteOpInterface.
  if (!allowRAR && !isa<AffineWriteOpInterface>(srcAccess.opInst) &&
      !isa<AffineWriteOpInterface>(dstAccess.opInst))
    return DependenceResult::NoDependence;

  // Create access relation from each MemRefAccess.
  FlatAffineRelation srcRel, dstRel;
  if (failed(srcAccess.getAccessRelation(srcRel)))
    return DependenceResult::Failure;
  if (failed(dstAccess.getAccessRelation(dstRel)))
    return DependenceResult::Failure;

  FlatAffineValueConstraints srcDomain = srcRel.getDomainSet();
  FlatAffineValueConstraints dstDomain = dstRel.getDomainSet();

  // Return 'NoDependence' if loopDepth > numCommonLoops and if the ancestor
  // operation of 'srcAccess' does not properly dominate the ancestor
  // operation of 'dstAccess' in the same common operation block.
  // Note: this check is skipped if 'allowRAR' is true, because because RAR
  // deps can exist irrespective of lexicographic ordering b/w src and dst.
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  assert(loopDepth <= numCommonLoops + 1);
  if (!allowRAR && loopDepth > numCommonLoops &&
      !srcAppearsBeforeDstInAncestralBlock(srcAccess, dstAccess, srcDomain,
                                           numCommonLoops)) {
    return DependenceResult::NoDependence;
  }

  // Compute the dependence relation by composing `srcRel` with the inverse of
  // `dstRel`. Doing this builds a relation between iteration domain of
  // `srcAccess` to the iteration domain of `dstAccess` which access the same
  // memory locations.
  dstRel.inverse();
  dstRel.compose(srcRel);
  *dependenceConstraints = dstRel;

  // Add 'src' happens before 'dst' ordering constraints.
  addOrderingConstraints(srcDomain, dstDomain, loopDepth,
                         dependenceConstraints);

  // Return 'NoDependence' if the solution space is empty: no dependence.
  if (dependenceConstraints->isEmpty())
    return DependenceResult::NoDependence;

  // Compute dependence direction vector and return true.
  if (dependenceComponents != nullptr)
    computeDirectionVector(srcDomain, dstDomain, loopDepth,
                           dependenceConstraints, dependenceComponents);

  LLVM_DEBUG(llvm::dbgs() << "Dependence polyhedron:\n");
  LLVM_DEBUG(dependenceConstraints->dump());
  return DependenceResult::HasDependence;
}

/// Gathers dependence components for dependences between all ops in loop nest
/// rooted at 'forOp' at loop depths in range [1, maxLoopDepth].
void mlir::getDependenceComponents(
    AffineForOp forOp, unsigned maxLoopDepth,
    std::vector<SmallVector<DependenceComponent, 2>> *depCompsVec) {
  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadAndStoreOps;
  forOp->walk([&](Operation *op) {
    if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op))
      loadAndStoreOps.push_back(op);
  });

  unsigned numOps = loadAndStoreOps.size();
  for (unsigned d = 1; d <= maxLoopDepth; ++d) {
    for (unsigned i = 0; i < numOps; ++i) {
      auto *srcOp = loadAndStoreOps[i];
      MemRefAccess srcAccess(srcOp);
      for (unsigned j = 0; j < numOps; ++j) {
        auto *dstOp = loadAndStoreOps[j];
        MemRefAccess dstAccess(dstOp);

        FlatAffineValueConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> depComps;
        // TODO: Explore whether it would be profitable to pre-compute and store
        // deps instead of repeatedly checking.
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints, &depComps);
        if (hasDependence(result))
          depCompsVec->push_back(depComps);
      }
    }
  }
}
