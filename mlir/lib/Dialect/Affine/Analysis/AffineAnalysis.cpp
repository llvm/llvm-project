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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "affine-analysis"

using namespace mlir;
using namespace presburger;

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
  std::optional<arith::AtomicRMWKind> maybeKind =
      TypeSwitch<Operation *, std::optional<arith::AtomicRMWKind>>(combinerOp)
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
          .Default([](Operation *) -> std::optional<arith::AtomicRMWKind> {
            // TODO: AtomicRMW supports other kinds of reductions this is
            // currently not detecting, add those when the need arises.
            return std::nullopt;
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

/// Returns true if `v` is allocated locally to `enclosingOp` -- i.e., it is
/// allocated by an operation nested within `enclosingOp`.
static bool isLocallyDefined(Value v, Operation *enclosingOp) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;

  if (hasSingleEffect<MemoryEffects::Allocate>(defOp, v) &&
      enclosingOp->isProperAncestor(defOp))
    return true;

  // Aliasing ops.
  auto viewOp = dyn_cast<ViewLikeOpInterface>(defOp);
  return viewOp && isLocallyDefined(viewOp.getViewSource(), enclosingOp);
}

bool mlir::isLoopMemoryParallel(AffineForOp forOp) {
  // Any memref-typed iteration arguments are treated as serializing.
  if (llvm::any_of(forOp.getResultTypes(),
                   [](Type type) { return type.isa<BaseMemRefType>(); }))
    return false;

  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadAndStoreOps;
  auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
      // Memrefs that are allocated inside `forOp` need not be considered.
      if (!isLocallyDefined(readOp.getMemRef(), forOp))
        loadAndStoreOps.push_back(op);
    } else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      // Filter out stores the same way as above.
      if (!isLocallyDefined(writeOp.getMemRef(), forOp))
        loadAndStoreOps.push_back(op);
    } else if (!isa<AffineForOp, AffineYieldOp, AffineIfOp>(op) &&
               !hasSingleEffect<MemoryEffects::Allocate>(op) &&
               !isMemoryEffectFree(op)) {
      // Alloc-like ops inside `forOp` are fine (they don't impact parallelism)
      // as long as they don't escape the loop (which has been checked above).
      return WalkResult::interrupt();
    }

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
      DependenceResult result =
          checkMemrefAccessDependence(srcAccess, dstAccess, depth);
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

// Builds a system of constraints with dimensional variables corresponding to
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
  SmallVector<Operation *, 8> loopOps;
  size_t numDims = 0;
  for (Operation *op : ops) {
    if (!isa<AffineForOp, AffineIfOp, AffineParallelOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "getIndexSet only handles affine.for/if/"
                                 "parallel ops");
      return failure();
    }
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      loopOps.push_back(forOp);
      // An AffineForOp retains only 1 induction variable.
      numDims += 1;
    } else if (AffineParallelOp parallelOp = dyn_cast<AffineParallelOp>(op)) {
      loopOps.push_back(parallelOp);
      numDims += parallelOp.getNumDims();
    }
  }
  extractInductionVars(loopOps, indices);
  // Reset while associating Values in 'indices' to the domain.
  *domain = FlatAffineValueConstraints(numDims, /*numSymbols=*/0,
                                       /*numLocals=*/0, indices);
  for (Operation *op : ops) {
    // Add constraints from forOp's bounds.
    if (AffineForOp forOp = dyn_cast<AffineForOp>(op)) {
      if (failed(domain->addAffineForOpDomain(forOp)))
        return failure();
    } else if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
      domain->addAffineIfOpDomain(ifOp);
    } else if (auto parallelOp = dyn_cast<AffineParallelOp>(op))
      if (failed(domain->addAffineParallelOpDomain(parallelOp)))
        return failure();
  }
  return success();
}

/// Computes the iteration domain for 'op' and populates 'indexSet', which
/// encapsulates the constraints involving loops surrounding 'op' and
/// potentially involving any Function symbols. The dimensional variables in
/// 'indexSet' correspond to the loops surrounding 'op' from outermost to
/// innermost.
static LogicalResult getOpIndexSet(Operation *op,
                                   FlatAffineValueConstraints *indexSet) {
  SmallVector<Operation *, 4> ops;
  getEnclosingAffineOps(*op, &ops);
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
      std::min(srcDomain.getNumDimVars(), dstDomain.getNumDimVars());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if ((!isAffineForInductionVar(srcDomain.getValue(i)) &&
         !isAffineParallelInductionVar(srcDomain.getValue(i))) ||
        (!isAffineForInductionVar(dstDomain.getValue(i)) &&
         !isAffineParallelInductionVar(dstDomain.getValue(i))) ||
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

/// Returns the closest surrounding block common to `opA` and `opB`. `opA` and
/// `opB` should be in the same affine scope. Returns nullptr if such a block
/// does not exist (when the two ops are in different blocks of an op starting
/// an `AffineScope`).
static Block *getCommonBlockInAffineScope(Operation *opA, Operation *opB) {
  // Get the chain of ancestor blocks for the given `MemRefAccess` instance. The
  // chain extends up to and includnig an op that starts an affine scope.
  auto getChainOfAncestorBlocks =
      [&](Operation *op, SmallVectorImpl<Block *> &ancestorBlocks) {
        Block *currBlock = op->getBlock();
        // Loop terminates when the currBlock is nullptr or its parent operation
        // holds an affine scope.
        while (currBlock &&
               !currBlock->getParentOp()->hasTrait<OpTrait::AffineScope>()) {
          ancestorBlocks.push_back(currBlock);
          currBlock = currBlock->getParentOp()->getBlock();
        }
        assert(currBlock &&
               "parent op starting an affine scope is always expected");
        ancestorBlocks.push_back(currBlock);
      };

  // Find the closest common block.
  SmallVector<Block *, 4> srcAncestorBlocks, dstAncestorBlocks;
  getChainOfAncestorBlocks(opA, srcAncestorBlocks);
  getChainOfAncestorBlocks(opB, dstAncestorBlocks);

  Block *commonBlock = nullptr;
  for (int i = srcAncestorBlocks.size() - 1, j = dstAncestorBlocks.size() - 1;
       i >= 0 && j >= 0 && srcAncestorBlocks[i] == dstAncestorBlocks[j];
       i--, j--)
    commonBlock = srcAncestorBlocks[i];

  return commonBlock;
}

/// Returns true if the ancestor operation of 'srcAccess' appears before the
/// ancestor operation of 'dstAccess' in their common ancestral block. The
/// operations for `srcAccess` and `dstAccess` are expected to be in the same
/// affine scope and have a common surrounding block within it.
static bool srcAppearsBeforeDstInAncestralBlock(const MemRefAccess &srcAccess,
                                                const MemRefAccess &dstAccess) {
  // Get Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
  Block *commonBlock =
      getCommonBlockInAffineScope(srcAccess.opInst, dstAccess.opInst);
  assert(commonBlock &&
         "ops expected to have a common surrounding block in affine scope");

  // Check the dominance relationship between the respective ancestors of the
  // src and dst in the Block of the innermost among the common loops.
  Operation *srcOp = commonBlock->findAncestorOpInBlock(*srcAccess.opInst);
  assert(srcOp && "src access op must lie in common block");
  Operation *dstOp = commonBlock->findAncestorOpInBlock(*dstAccess.opInst);
  assert(dstOp && "dest access op must lie in common block");

  // Determine whether dstOp comes after srcOp.
  return srcOp->isBeforeInBlock(dstOp);
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
  unsigned numSrcDims = srcDomain.getNumDimVars();
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
  unsigned numIdsToEliminate = dependenceDomain->getNumVars();
  // Add new variables to 'dependenceDomain' to represent the direction
  // constraints for each shared loop.
  dependenceDomain->insertDimVar(/*pos=*/0, /*num=*/numCommonLoops);

  // Add equality constraints for each common loop, setting newly introduced
  // variable at column 'j' to the 'dst' IV minus the 'src IV.
  SmallVector<int64_t, 4> eq;
  eq.resize(dependenceDomain->getNumCols());
  unsigned numSrcDims = srcDomain.getNumDimVars();
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
        dependenceDomain->getConstantBound64(IntegerPolyhedron::LB, j);
    (*dependenceComponents)[j].lb =
        lbConst.value_or(std::numeric_limits<int64_t>::min());
    auto ubConst =
        dependenceDomain->getConstantBound64(IntegerPolyhedron::UB, j);
    (*dependenceComponents)[j].ub =
        ubConst.value_or(std::numeric_limits<int64_t>::max());
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
  for (unsigned i = 0, e = domain.getNumDimVars(); i < e; ++i) {
    unsigned loc;
    if (rel.findVar(domain.getValue(i), &loc)) {
      rel.swapVar(i, loc);
    } else {
      rel.insertDomainVar(i);
      rel.setValue(i, domain.getValue(i));
    }
  }

  // Append domain constraints to `rel`.
  domainRel.appendRangeVar(rel.getNumRangeDims());
  domainRel.mergeSymbolVars(rel);
  domainRel.mergeLocalVars(rel);
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
//  [src-dim-variables, dst-dim-variables, symbols, constant]
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
  LLVM_DEBUG(srcAccess.opInst->dump());
  LLVM_DEBUG(dstAccess.opInst->dump());

  // Return 'NoDependence' if these accesses do not access the same memref.
  if (srcAccess.memref != dstAccess.memref)
    return DependenceResult::NoDependence;

  // Return 'NoDependence' if one of these accesses is not an
  // AffineWriteOpInterface.
  if (!allowRAR && !isa<AffineWriteOpInterface>(srcAccess.opInst) &&
      !isa<AffineWriteOpInterface>(dstAccess.opInst))
    return DependenceResult::NoDependence;

  // We can't analyze further if the ops lie in different affine scopes or have
  // no common block in an affine scope.
  if (getAffineScope(srcAccess.opInst) != getAffineScope(dstAccess.opInst))
    return DependenceResult::Failure;
  if (!getCommonBlockInAffineScope(srcAccess.opInst, dstAccess.opInst))
    return DependenceResult::Failure;

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
      !srcAppearsBeforeDstInAncestralBlock(srcAccess, dstAccess)) {
    return DependenceResult::NoDependence;
  }

  // Compute the dependence relation by composing `srcRel` with the inverse of
  // `dstRel`. Doing this builds a relation between iteration domain of
  // `srcAccess` to the iteration domain of `dstAccess` which access the same
  // memory locations.
  dstRel.inverse();
  dstRel.compose(srcRel);

  // Add 'src' happens before 'dst' ordering constraints.
  addOrderingConstraints(srcDomain, dstDomain, loopDepth, &dstRel);

  // Return 'NoDependence' if the solution space is empty: no dependence.
  if (dstRel.isEmpty())
    return DependenceResult::NoDependence;

  // Compute dependence direction vector and return true.
  if (dependenceComponents != nullptr)
    computeDirectionVector(srcDomain, dstDomain, loopDepth, &dstRel,
                           dependenceComponents);

  LLVM_DEBUG(llvm::dbgs() << "Dependence polyhedron:\n");
  LLVM_DEBUG(dstRel.dump());

  if (dependenceConstraints)
    *dependenceConstraints = dstRel;
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

        SmallVector<DependenceComponent, 2> depComps;
        // TODO: Explore whether it would be profitable to pre-compute and store
        // deps instead of repeatedly checking.
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, /*dependenceConstraints=*/nullptr,
            &depComps);
        if (hasDependence(result))
          depCompsVec->push_back(depComps);
      }
    }
  }
}
