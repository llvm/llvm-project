//===- Utils.h - General analysis utilities ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_ANALYSIS_UTILS_H
#define MLIR_DIALECT_AFFINE_ANALYSIS_UTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>

namespace mlir {

class AffineForOp;
class AffineValueMap;
class Block;
class Location;
struct MemRefAccess;
class Operation;
class Value;

/// Populates 'loops' with IVs of the affine.for ops surrounding 'op' ordered
/// from the outermost 'affine.for' operation to the innermost one.
void getAffineForIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops);

/// Populates 'ops' with affine operations enclosing `op` ordered from outermost
/// to innermost. affine.for, affine.if, or affine.parallel ops comprise such
/// surrounding affine ops.
/// TODO: Change this to return a list of enclosing ops up until the op that
/// starts an `AffineScope`. In such a case, `ops` is guaranteed by design to
/// have a successive chain of affine parent ops, and this is primarily what is
/// needed for most analyses.
void getEnclosingAffineOps(Operation &op, SmallVectorImpl<Operation *> *ops);

/// Returns the nesting depth of this operation, i.e., the number of loops
/// surrounding this operation.
unsigned getNestingDepth(Operation *op);

/// Returns whether a loop is a parallel loop and contains a reduction loop.
bool isLoopParallelAndContainsReduction(AffineForOp forOp);

/// Returns in 'sequentialLoops' all sequential loops in loop nest rooted
/// at 'forOp'.
void getSequentialLoops(AffineForOp forOp,
                        llvm::SmallDenseSet<Value, 8> *sequentialLoops);

/// Enumerates different result statuses of slice computation by
/// `computeSliceUnion`
// TODO: Identify and add different kinds of failures during slice computation.
struct SliceComputationResult {
  enum ResultEnum {
    Success,
    IncorrectSliceFailure, // Slice is computed, but it is incorrect.
    GenericFailure,        // Unable to compute src loop computation slice.
  } value;
  SliceComputationResult(ResultEnum v) : value(v) {}
};

/// ComputationSliceState aggregates loop IVs, loop bound AffineMaps and their
/// associated operands for a set of loops within a loop nest (typically the
/// set of loops surrounding a store operation). Loop bound AffineMaps which
/// are non-null represent slices of that loop's iteration space.
struct ComputationSliceState {
  // List of sliced loop IVs (ordered from outermost to innermost).
  // EX: 'ivs[i]' has lower bound 'lbs[i]' and upper bound 'ubs[i]'.
  SmallVector<Value, 4> ivs;
  // List of lower bound AffineMaps.
  SmallVector<AffineMap, 4> lbs;
  // List of upper bound AffineMaps.
  SmallVector<AffineMap, 4> ubs;
  // List of lower bound operands (lbOperands[i] are used by 'lbs[i]').
  std::vector<SmallVector<Value, 4>> lbOperands;
  // List of upper bound operands (ubOperands[i] are used by 'ubs[i]').
  std::vector<SmallVector<Value, 4>> ubOperands;
  // Slice loop nest insertion point in target loop nest.
  Block::iterator insertPoint;
  // Adds to 'cst' with constraints which represent the slice bounds on 'ivs'
  // in 'this'. Specifically, the values in 'ivs' are added to 'cst' as dim
  // variables and the values in 'lb/ubOperands' are added as symbols.
  // Constraints are added for all loop IV bounds (dim or symbol), and
  // constraints are added for slice bounds in 'lbs'/'ubs'.
  // Returns failure if we cannot add loop bounds because of unsupported cases.
  LogicalResult getAsConstraints(FlatAffineValueConstraints *cst);

  /// Adds to 'cst' constraints which represent the original loop bounds on
  /// 'ivs' in 'this'. This corresponds to the original domain of the loop nest
  /// from which the slice is being computed. Returns failure if we cannot add
  /// loop bounds because of unsupported cases.
  LogicalResult getSourceAsConstraints(FlatAffineValueConstraints &cst);

  // Clears all bounds and operands in slice state.
  void clearBounds();

  /// Returns true if the computation slice is empty.
  bool isEmpty() const { return ivs.empty(); }

  /// Returns true if the computation slice encloses all the iterations of the
  /// sliced loop nest. Returns false if it does not. Returns std::nullopt if it
  /// cannot determine if the slice is maximal or not.
  // TODO: Cache 'isMaximal' so that we don't recompute it when the slice
  // information hasn't changed.
  Optional<bool> isMaximal() const;

  /// Checks the validity of the slice computed. This is done using the
  /// following steps:
  /// 1. Get the new domain of the slice that would be created if fusion
  /// succeeds. This domain gets constructed with source loop IVS and
  /// destination loop IVS as dimensions.
  /// 2. Project out the dimensions of the destination loop from the domain
  /// above calculated in step(1) to express it purely in terms of the source
  /// loop IVs.
  /// 3. Calculate a set difference between the iterations of the new domain and
  /// the original domain of the source loop.
  /// If this difference is empty, the slice is declared to be valid. Otherwise,
  /// return false as it implies that the effective fusion results in at least
  /// one iteration of the slice that was not originally in the source's domain.
  /// If the validity cannot be determined, returns std::nullopt.
  Optional<bool> isSliceValid();

  void dump() const;

private:
  /// Fast check to determine if the computation slice is maximal. Returns true
  /// if each slice dimension maps to an existing dst dimension and both the src
  /// and the dst loops for those dimensions have the same bounds. Returns false
  /// if both the src and the dst loops don't have the same bounds. Returns
  /// std::nullopt if none of the above can be proven.
  Optional<bool> isSliceMaximalFastCheck() const;
};

/// Computes the computation slice loop bounds for one loop nest as affine maps
/// of the other loop nest's IVs and symbols, using 'dependenceConstraints'
/// computed between 'depSourceAccess' and 'depSinkAccess'.
/// If 'isBackwardSlice' is true, a backwards slice is computed in which the
/// slice bounds of loop nest surrounding 'depSourceAccess' are computed in
/// terms of loop IVs and symbols of the loop nest surrounding 'depSinkAccess'
/// at 'loopDepth'.
/// If 'isBackwardSlice' is false, a forward slice is computed in which the
/// slice bounds of loop nest surrounding 'depSinkAccess' are computed in terms
/// of loop IVs and symbols of the loop nest surrounding 'depSourceAccess' at
/// 'loopDepth'.
/// The slice loop bounds and associated operands are returned in 'sliceState'.
//
//  Backward slice example:
//
//    affine.for %i0 = 0 to 10 {
//      affine.store %cst, %0[%i0] : memref<100xf32>  // 'depSourceAccess'
//    }
//    affine.for %i1 = 0 to 10 {
//      %v = affine.load %0[%i1] : memref<100xf32>    // 'depSinkAccess'
//    }
//
//    // Backward computation slice of loop nest '%i0'.
//    affine.for %i0 = (d0) -> (d0)(%i1) to (d0) -> (d0 + 1)(%i1) {
//      affine.store %cst, %0[%i0] : memref<100xf32>  // 'depSourceAccess'
//    }
//
//  Forward slice example:
//
//    affine.for %i0 = 0 to 10 {
//      affine.store %cst, %0[%i0] : memref<100xf32>  // 'depSourceAccess'
//    }
//    affine.for %i1 = 0 to 10 {
//      %v = affine.load %0[%i1] : memref<100xf32>    // 'depSinkAccess'
//    }
//
//    // Forward computation slice of loop nest '%i1'.
//    affine.for %i1 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 1)(%i0) {
//      %v = affine.load %0[%i1] : memref<100xf32>    // 'depSinkAccess'
//    }
//
void getComputationSliceState(Operation *depSourceOp, Operation *depSinkOp,
                              FlatAffineValueConstraints *dependenceConstraints,
                              unsigned loopDepth, bool isBackwardSlice,
                              ComputationSliceState *sliceState);

/// Return the number of iterations for the `slicetripCountMap` provided.
uint64_t getSliceIterationCount(
    const llvm::SmallDenseMap<Operation *, uint64_t, 8> &sliceTripCountMap);

/// Builds a map 'tripCountMap' from AffineForOp to constant trip count for
/// loop nest surrounding represented by slice loop bounds in 'slice'. Returns
/// true on success, false otherwise (if a non-constant trip count was
/// encountered).
bool buildSliceTripCountMap(
    const ComputationSliceState &slice,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountMap);

/// Computes in 'sliceUnion' the union of all slice bounds computed at
/// 'loopDepth' between all dependent pairs of ops in 'opsA' and 'opsB', and
/// then verifies if it is valid. The parameter 'numCommonLoops' is the number
/// of loops common to the operations in 'opsA' and 'opsB'. If 'isBackwardSlice'
/// is true, computes slice bounds for loop nest surrounding ops in 'opsA', as a
/// function of IVs and symbols of loop nest surrounding ops in 'opsB' at
/// 'loopDepth'. If 'isBackwardSlice' is false, computes slice bounds for loop
/// nest surrounding ops in 'opsB', as a function of IVs and symbols of loop
/// nest surrounding ops in 'opsA' at 'loopDepth'. Returns
/// 'SliceComputationResult::Success' if union was computed correctly, an
/// appropriate 'failure' otherwise.
// TODO: Change this API to take 'forOpA'/'forOpB'.
SliceComputationResult
computeSliceUnion(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                  unsigned loopDepth, unsigned numCommonLoops,
                  bool isBackwardSlice, ComputationSliceState *sliceUnion);

/// Creates a clone of the computation contained in the loop nest surrounding
/// 'srcOpInst', slices the iteration space of src loop based on slice bounds
/// in 'sliceState', and inserts the computation slice at the beginning of the
/// operation block of the loop at 'dstLoopDepth' in the loop nest surrounding
/// 'dstOpInst'. Returns the top-level loop of the computation slice on
/// success, returns nullptr otherwise.
// Loop depth is a crucial optimization choice that determines where to
// materialize the results of the backward slice - presenting a trade-off b/w
// storage and redundant computation in several cases.
// TODO: Support computation slices with common surrounding loops.
AffineForOp insertBackwardComputationSlice(Operation *srcOpInst,
                                           Operation *dstOpInst,
                                           unsigned dstLoopDepth,
                                           ComputationSliceState *sliceState);

/// A region of a memref's data space; this is typically constructed by
/// analyzing load/store op's on this memref and the index space of loops
/// surrounding such op's.
// For example, the memref region for a load operation at loop depth = 1:
//
//    affine.for %i = 0 to 32 {
//      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        affine.load %A[%ii]
//      }
//    }
//
// Region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineValueConstraints symbolic in %i.
//
struct MemRefRegion {
  explicit MemRefRegion(Location loc) : loc(loc) {}

  /// Computes the memory region accessed by this memref with the region
  /// represented as constraints symbolic/parametric in 'loopDepth' loops
  /// surrounding opInst. The computed region's 'cst' field has exactly as many
  /// dimensional variables as the rank of the memref, and *potentially*
  /// additional symbolic variables which could include any of the loop IVs
  /// surrounding opInst up until 'loopDepth' and another additional Function
  /// symbols involved with the access (for eg., those appear in affine.apply's,
  /// loop bounds, etc.). If 'sliceState' is non-null, operands from
  /// 'sliceState' are added as symbols, and the following constraints are added
  /// to the system:
  /// *) Inequality constraints which represent loop bounds for 'sliceState'
  ///    operands which are loop IVS (these represent the destination loop IVs
  ///    of the slice, and are added as symbols to MemRefRegion's constraint
  ///    system).
  /// *) Inequality constraints for the slice bounds in 'sliceState', which
  ///    represent the bounds on the loop IVs in this constraint system w.r.t
  ///    to slice operands (which correspond to symbols).
  /// If 'addMemRefDimBounds' is true, constant upper/lower bounds
  /// [0, memref.getDimSize(i)) are added for each MemRef dimension 'i'.
  ///
  ///  For example, the memref region for this operation at loopDepth = 1 will
  ///  be:
  ///
  ///    affine.for %i = 0 to 32 {
  ///      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
  ///        load %A[%ii]
  ///      }
  ///    }
  ///
  ///   {memref = %A, write = false, {%i <= m0 <= %i + 7} }
  /// The last field is a 2-d FlatAffineValueConstraints symbolic in %i.
  ///
  LogicalResult compute(Operation *op, unsigned loopDepth,
                        const ComputationSliceState *sliceState = nullptr,
                        bool addMemRefDimBounds = true);

  FlatAffineValueConstraints *getConstraints() { return &cst; }
  const FlatAffineValueConstraints *getConstraints() const { return &cst; }
  bool isWrite() const { return write; }
  void setWrite(bool flag) { write = flag; }

  /// Returns a constant upper bound on the number of elements in this region if
  /// bounded by a known constant (always possible for static shapes),
  /// std::nullopt otherwise. Note that the symbols of the region are treated
  /// specially, i.e., the returned bounding constant holds for *any given*
  /// value of the symbol variables. The 'shape' vector is set to the
  /// corresponding dimension-wise bounds major to minor. The number of elements
  /// and all the dimension-wise bounds are guaranteed to be non-negative. We
  /// use int64_t instead of uint64_t since index types can be at most
  /// int64_t. `lbs` are set to the lower bounds for each of the rank
  /// dimensions, and lbDivisors contains the corresponding denominators for
  /// floorDivs.
  Optional<int64_t> getConstantBoundingSizeAndShape(
      SmallVectorImpl<int64_t> *shape = nullptr,
      std::vector<SmallVector<int64_t, 4>> *lbs = nullptr,
      SmallVectorImpl<int64_t> *lbDivisors = nullptr) const;

  /// Gets the lower and upper bound map for the dimensional variable at
  /// `pos`.
  void getLowerAndUpperBound(unsigned pos, AffineMap &lbMap,
                             AffineMap &ubMap) const;

  /// A wrapper around FlatAffineValueConstraints::getConstantBoundOnDimSize().
  /// 'pos' corresponds to the position of the memref shape's dimension (major
  /// to minor) which matches 1:1 with the dimensional variable positions in
  /// 'cst'.
  Optional<int64_t>
  getConstantBoundOnDimSize(unsigned pos,
                            SmallVectorImpl<int64_t> *lb = nullptr,
                            int64_t *lbFloorDivisor = nullptr) const {
    assert(pos < getRank() && "invalid position");
    return cst.getConstantBoundOnDimSize64(pos, lb);
  }

  /// Returns the size of this MemRefRegion in bytes.
  Optional<int64_t> getRegionSize();

  // Wrapper around FlatAffineValueConstraints::unionBoundingBox.
  LogicalResult unionBoundingBox(const MemRefRegion &other);

  /// Returns the rank of the memref that this region corresponds to.
  unsigned getRank() const;

  /// Memref that this region corresponds to.
  Value memref;

  /// Read or write.
  bool write = false;

  /// If there is more than one load/store op associated with the region, the
  /// location information would correspond to one of those op's.
  Location loc;

  /// Region (data space) of the memref accessed. This set will thus have at
  /// least as many dimensional variables as the shape dimensionality of the
  /// memref, and these are the leading dimensions of the set appearing in that
  /// order (major to minor / outermost to innermost). There may be additional
  /// variables since getMemRefRegion() is called with a specific loop depth,
  /// and thus the region is symbolic in the outer surrounding loops at that
  /// depth.
  // TODO: Replace this to exploit HyperRectangularSet.
  FlatAffineValueConstraints cst;
};

/// Returns the size of memref data in bytes if it's statically shaped,
/// std::nullopt otherwise.
Optional<uint64_t> getMemRefSizeInBytes(MemRefType memRefType);

/// Checks a load or store op for an out of bound access; returns failure if the
/// access is out of bounds along any of the dimensions, success otherwise.
/// Emits a diagnostic error (with location information) if emitError is true.
template <typename LoadOrStoreOpPointer>
LogicalResult boundCheckLoadOrStoreOp(LoadOrStoreOpPointer loadOrStoreOp,
                                      bool emitError = true);

/// Returns the number of surrounding loops common to both A and B.
unsigned getNumCommonSurroundingLoops(Operation &a, Operation &b);

/// Gets the memory footprint of all data touched in the specified memory space
/// in bytes; if the memory space is unspecified, considers all memory spaces.
Optional<int64_t> getMemoryFootprintBytes(AffineForOp forOp,
                                          int memorySpace = -1);

/// Simplify the integer set by simplifying the underlying affine expressions by
/// flattening and some simple inference. Also, drop any duplicate constraints.
/// Returns the simplified integer set. This method runs in time linear in the
/// number of constraints.
IntegerSet simplifyIntegerSet(IntegerSet set);

/// Returns the innermost common loop depth for the set of operations in 'ops'.
unsigned getInnermostCommonLoopDepth(
    ArrayRef<Operation *> ops,
    SmallVectorImpl<AffineForOp> *surroundingLoops = nullptr);

/// Try to simplify the given affine.min or affine.max op to an affine map with
/// a single result and operands, taking into account the specified constraint
/// set. Return failure if no simplified version could be found.
FailureOr<AffineValueMap>
simplifyConstrainedMinMaxOp(Operation *op,
                            FlatAffineValueConstraints constraints);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_ANALYSIS_UTILS_H
