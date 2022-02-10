//===- Utils.h - Affine dialect utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares a set of utilities for the affine dialect ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AFFINE_UTILS_H
#define MLIR_DIALECT_AFFINE_UTILS_H

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"

namespace mlir {

class AffineForOp;
class AffineIfOp;
class AffineParallelOp;
class DominanceInfo;
class Operation;
class PostDominanceInfo;

namespace memref {
class AllocOp;
} // namespace memref

struct LogicalResult;

using ReductionLoopMap = DenseMap<Operation *, SmallVector<LoopReduction, 2>>;

/// Replaces a parallel affine.for op with a 1-d affine.parallel op. `forOp`'s
/// body is taken by the affine.parallel op and the former is erased.
/// (mlir::isLoopParallel can be used to detect a parallel affine.for op.) The
/// reductions specified in `parallelReductions` are also parallelized.
/// Parallelization will fail in the presence of loop iteration arguments that
/// are not listed in `parallelReductions`.
LogicalResult
affineParallelize(AffineForOp forOp,
                  ArrayRef<LoopReduction> parallelReductions = {});

/// Hoists out affine.if/else to as high as possible, i.e., past all invariant
/// affine.fors/parallel's. Returns success if any hoisting happened; folded` is
/// set to true if the op was folded or erased. This hoisting could lead to
/// significant code expansion in some cases.
LogicalResult hoistAffineIfOp(AffineIfOp ifOp, bool *folded = nullptr);

/// Holds parameters to perform n-D vectorization on a single loop nest.
/// For example, for the following loop nest:
///
/// func @vec2d(%in: memref<64x128x512xf32>, %out: memref<64x128x512xf32>) {
///   affine.for %i0 = 0 to 64 {
///     affine.for %i1 = 0 to 128 {
///       affine.for %i2 = 0 to 512 {
///         %ld = affine.load %in[%i0, %i1, %i2] : memref<64x128x512xf32>
///         affine.store %ld, %out[%i0, %i1, %i2] : memref<64x128x512xf32>
///       }
///     }
///   }
///   return
/// }
///
/// and VectorizationStrategy = 'vectorSizes = {8, 4}', 'loopToVectorDim =
/// {{i1->0}, {i2->1}}', SuperVectorizer will generate:
///
///  func @vec2d(%arg0: memref<64x128x512xf32>, %arg1: memref<64x128x512xf32>) {
///    affine.for %arg2 = 0 to 64 {
///      affine.for %arg3 = 0 to 128 step 8 {
///        affine.for %arg4 = 0 to 512 step 4 {
///          %cst = arith.constant 0.000000e+00 : f32
///          %0 = vector.transfer_read %arg0[%arg2, %arg3, %arg4], %cst : ...
///          vector.transfer_write %0, %arg1[%arg2, %arg3, %arg4] : ...
///        }
///      }
///    }
///    return
///  }
// TODO: Hoist to a VectorizationStrategy.cpp when appropriate.
struct VectorizationStrategy {
  // Vectorization factors to apply to each target vector dimension.
  // Each factor will be applied to a different loop.
  SmallVector<int64_t, 8> vectorSizes;
  // Maps each AffineForOp vectorization candidate with its vector dimension.
  // The candidate will be vectorized using the vectorization factor in
  // 'vectorSizes' for that dimension.
  DenseMap<Operation *, unsigned> loopToVectorDim;
  // Maps loops that implement vectorizable reductions to the corresponding
  // reduction descriptors.
  ReductionLoopMap reductionLoops;
};

/// Replace affine store and load accesses by scalars by forwarding stores to
/// loads and eliminate invariant affine loads; consequently, eliminate dead
/// allocs.
void affineScalarReplace(FuncOp f, DominanceInfo &domInfo,
                         PostDominanceInfo &postDomInfo);

/// Vectorizes affine loops in 'loops' using the n-D vectorization factors in
/// 'vectorSizes'. By default, each vectorization factor is applied
/// inner-to-outer to the loops of each loop nest. 'fastestVaryingPattern' can
/// be optionally used to provide a different loop vectorization order.
/// If `reductionLoops` is not empty, the given reduction loops may be
/// vectorized along the reduction dimension.
/// TODO: Vectorizing reductions is supported only for 1-D vectorization.
void vectorizeAffineLoops(
    Operation *parentOp,
    llvm::DenseSet<Operation *, DenseMapInfo<Operation *>> &loops,
    ArrayRef<int64_t> vectorSizes, ArrayRef<int64_t> fastestVaryingPattern,
    const ReductionLoopMap &reductionLoops = ReductionLoopMap());

/// External utility to vectorize affine loops from a single loop nest using an
/// n-D vectorization strategy (see doc in VectorizationStrategy definition).
/// Loops are provided in a 2D vector container. The first dimension represents
/// the nesting level relative to the loops to be vectorized. The second
/// dimension contains the loops. This means that:
///   a) every loop in 'loops[i]' must have a parent loop in 'loops[i-1]',
///   b) a loop in 'loops[i]' may or may not have a child loop in 'loops[i+1]'.
///
/// For example, for the following loop nest:
///
///   func @vec2d(%in0: memref<64x128x512xf32>, %in1: memref<64x128x128xf32>,
///               %out0: memref<64x128x512xf32>,
///               %out1: memref<64x128x128xf32>) {
///     affine.for %i0 = 0 to 64 {
///       affine.for %i1 = 0 to 128 {
///         affine.for %i2 = 0 to 512 {
///           %ld = affine.load %in0[%i0, %i1, %i2] : memref<64x128x512xf32>
///           affine.store %ld, %out0[%i0, %i1, %i2] : memref<64x128x512xf32>
///         }
///         affine.for %i3 = 0 to 128 {
///           %ld = affine.load %in1[%i0, %i1, %i3] : memref<64x128x128xf32>
///           affine.store %ld, %out1[%i0, %i1, %i3] : memref<64x128x128xf32>
///         }
///       }
///     }
///     return
///   }
///
/// loops = {{%i0}, {%i2, %i3}}, to vectorize the outermost and the two
/// innermost loops;
/// loops = {{%i1}, {%i2, %i3}}, to vectorize the middle and the two innermost
/// loops;
/// loops = {{%i2}}, to vectorize only the first innermost loop;
/// loops = {{%i3}}, to vectorize only the second innermost loop;
/// loops = {{%i1}}, to vectorize only the middle loop.
LogicalResult
vectorizeAffineLoopNest(std::vector<SmallVector<AffineForOp, 2>> &loops,
                        const VectorizationStrategy &strategy);

/// Normalize a affine.parallel op so that lower bounds are 0 and steps are 1.
/// As currently implemented, this transformation cannot fail and will return
/// early if the op is already in a normalized form.
void normalizeAffineParallel(AffineParallelOp op);

/// Normalize an affine.for op. If the affine.for op has only a single iteration
/// only then it is simply promoted, else it is normalized in the traditional
/// way, by converting the lower bound to zero and loop step to one. The upper
/// bound is set to the trip count of the loop. For now, original loops must
/// have lower bound with a single result only. There is no such restriction on
/// upper bounds.
void normalizeAffineFor(AffineForOp op);

/// Traverse `e` and return an AffineExpr where all occurrences of `dim` have
/// been replaced by either:
///  - `min` if `positivePath` is true when we reach an occurrence of `dim`
///  - `max` if `positivePath` is true when we reach an occurrence of `dim`
/// `positivePath` is negated each time we hit a multiplicative or divisive
/// binary op with a constant negative coefficient.
AffineExpr substWithMin(AffineExpr e, AffineExpr dim, AffineExpr min,
                        AffineExpr max, bool positivePath = true);

/// Replaces all "dereferencing" uses of `oldMemRef` with `newMemRef` while
/// optionally remapping the old memref's indices using the supplied affine map,
/// `indexRemap`. The new memref could be of a different shape or rank.
/// `extraIndices` provides any additional access indices to be added to the
/// start.
///
/// `indexRemap` remaps indices of the old memref access to a new set of indices
/// that are used to index the memref. Additional input operands to indexRemap
/// can be optionally provided in `extraOperands`, and they occupy the start
/// of its input list. `indexRemap`'s dimensional inputs are expected to
/// correspond to memref's indices, and its symbolic inputs if any should be
/// provided in `symbolOperands`.
///
/// `domOpFilter`, if non-null, restricts the replacement to only those
/// operations that are dominated by the former; similarly, `postDomOpFilter`
/// restricts replacement to only those operations that are postdominated by it.
///
/// 'allowNonDereferencingOps', if set, allows replacement of non-dereferencing
/// uses of a memref without any requirement for access index rewrites as long
/// as the user operation has the MemRefsNormalizable trait. The default value
/// of this flag is false.
///
/// 'replaceInDeallocOp', if set, lets DeallocOp, a non-dereferencing user, to
/// also be a candidate for replacement. The default value of this flag is
/// false.
///
/// Returns true on success and false if the replacement is not possible,
/// whenever a memref is used as an operand in a non-dereferencing context and
/// 'allowNonDereferencingOps' is false, except for dealloc's on the memref
/// which are left untouched. See comments at function definition for an
/// example.
//
//  Ex: to replace load %A[%i, %j] with load %Abuf[%t mod 2, %ii - %i, %j]:
//  The SSA value corresponding to '%t mod 2' should be in 'extraIndices', and
//  index remap will perform (%i, %j) -> (%ii - %i, %j), i.e., indexRemap = (d0,
//  d1, d2) -> (d0 - d1, d2), and %ii will be the extra operand. Without any
//  extra operands, note that 'indexRemap' would just be applied to existing
//  indices (%i, %j).
//  TODO: allow extraIndices to be added at any position.
LogicalResult replaceAllMemRefUsesWith(
    Value oldMemRef, Value newMemRef, ArrayRef<Value> extraIndices = {},
    AffineMap indexRemap = AffineMap(), ArrayRef<Value> extraOperands = {},
    ArrayRef<Value> symbolOperands = {}, Operation *domOpFilter = nullptr,
    Operation *postDomOpFilter = nullptr, bool allowNonDereferencingOps = false,
    bool replaceInDeallocOp = false);

/// Performs the same replacement as the other version above but only for the
/// dereferencing uses of `oldMemRef` in `op`, except in cases where
/// 'allowNonDereferencingOps' is set to true where we replace the
/// non-dereferencing uses as well.
LogicalResult replaceAllMemRefUsesWith(Value oldMemRef, Value newMemRef,
                                       Operation *op,
                                       ArrayRef<Value> extraIndices = {},
                                       AffineMap indexRemap = AffineMap(),
                                       ArrayRef<Value> extraOperands = {},
                                       ArrayRef<Value> symbolOperands = {},
                                       bool allowNonDereferencingOps = false);

/// Rewrites the memref defined by this alloc op to have an identity layout map
/// and updates all its indexing uses. Returns failure if any of its uses
/// escape (while leaving the IR in a valid state).
LogicalResult normalizeMemRef(memref::AllocOp *op);

/// Uses the old memref type map layout and computes the new memref type to have
/// a new shape and a layout map, where the old layout map has been normalized
/// to an identity layout map. It returns the old memref in case no
/// normalization was needed or a failure occurs while transforming the old map
/// layout to an identity layout map.
MemRefType normalizeMemRefType(MemRefType memrefType, OpBuilder builder,
                               unsigned numSymbolicOperands);

/// Creates and inserts into 'builder' a new AffineApplyOp, with the number of
/// its results equal to the number of operands, as a composition
/// of all other AffineApplyOps reachable from input parameter 'operands'. If
/// different operands were drawing results from multiple affine apply ops,
/// these will also be collected into a single (multi-result) affine apply op.
/// The final results of the composed AffineApplyOp are returned in output
/// parameter 'results'. Returns the affine apply op created.
Operation *createComposedAffineApplyOp(OpBuilder &builder, Location loc,
                                       ArrayRef<Value> operands,
                                       ArrayRef<Operation *> affineApplyOps,
                                       SmallVectorImpl<Value> *results);

/// Given an operation, inserts one or more single result affine apply
/// operations, results of which are exclusively used by this operation.
/// The operands of these newly created affine apply ops are
/// guaranteed to be loop iterators or terminal symbols of a function.
///
/// Before
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %v = "compute"(%idx, ...)
///
/// After
///
/// affine.for %i = 0 to #map(%N)
///   %idx = affine.apply (d0) -> (d0 mod 2) (%i)
///   send %A[%idx], ...
///   %idx_ = affine.apply (d0) -> (d0 mod 2) (%i)
///   %v = "compute"(%idx_, ...)

/// This allows the application of different transformations on send and
/// compute (for eg. different shifts/delays)
///
/// Fills `sliceOps` with the list of affine.apply operations.
/// In the following cases, `sliceOps` remains empty:
///   1. If none of opInst's operands were the result of an affine.apply
///      (i.e., there was no affine computation slice to create).
///   2. If all the affine.apply op's supplying operands to this opInst did not
///      have any uses other than those in this opInst.
void createAffineComputationSlice(Operation *opInst,
                                  SmallVectorImpl<AffineApplyOp> *sliceOps);

/// Emit code that computes the given affine expression using standard
/// arithmetic operations applied to the provided dimension and symbol values.
Value expandAffineExpr(OpBuilder &builder, Location loc, AffineExpr expr,
                       ValueRange dimValues, ValueRange symbolValues);

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
Optional<SmallVector<Value, 8>> expandAffineMap(OpBuilder &builder,
                                                Location loc,
                                                AffineMap affineMap,
                                                ValueRange operands);

} // namespace mlir

#endif // MLIR_DIALECT_AFFINE_UTILS_H
