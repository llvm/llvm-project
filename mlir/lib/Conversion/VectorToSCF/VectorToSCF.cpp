//===- VectorToSCF.cpp - Conversion from Vector to mix of SCF and Std -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-dependent lowering of vector transfer operations.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using vector::TransferReadOp;
using vector::TransferWriteOp;

/// Helper class captures the common information needed to lower N>1-D vector
/// transfer operations (read and write).
/// On construction, this class opens an edsc::ScopedContext for simpler IR
/// manipulation.
/// In pseudo-IR, for an n-D vector_transfer_read such as:
///
/// ```
///   vector_transfer_read(%m, %offsets, identity_map, %fill) :
///     memref<(leading_dims) x (major_dims) x (minor_dims) x type>,
///     vector<(major_dims) x (minor_dims) x type>
/// ```
///
/// where rank(minor_dims) is the lower-level vector rank (e.g. 1 for LLVM or
/// higher).
///
/// This is the entry point to emitting pseudo-IR resembling:
///
/// ```
///   %tmp = alloc(): memref<(major_dims) x vector<minor_dim x type>>
///   for (%ivs_major, {0}, {vector_shape}, {1}) { // (N-1)-D loop nest
///     if (any_of(%ivs_major + %offsets, <, major_dims)) {
///       %v = vector_transfer_read(
///         {%offsets_leading, %ivs_major + %offsets_major, %offsets_minor},
///          %ivs_minor):
///         memref<(leading_dims) x (major_dims) x (minor_dims) x type>,
///         vector<(minor_dims) x type>;
///       store(%v, %tmp);
///     } else {
///       %v = splat(vector<(minor_dims) x type>, %fill)
///       store(%v, %tmp, %ivs_major);
///     }
///   }
///   %res = load(%tmp, %0): memref<(major_dims) x vector<minor_dim x type>>):
//      vector<(major_dims) x (minor_dims) x type>
/// ```
///
template <typename ConcreteOp>
class NDTransferOpHelper {
public:
  NDTransferOpHelper(PatternRewriter &rewriter, ConcreteOp xferOp)
      : rewriter(rewriter), loc(xferOp.getLoc()),
        scope(std::make_unique<ScopedContext>(rewriter, loc)), xferOp(xferOp),
        op(xferOp.getOperation()) {
    vectorType = xferOp.getVectorType();
    // TODO(ntv, ajcbik): when we go to k > 1-D vectors adapt minorRank.
    minorRank = 1;
    majorRank = vectorType.getRank() - minorRank;
    leadingRank = xferOp.getMemRefType().getRank() - (majorRank + minorRank);
    majorVectorType =
        VectorType::get(vectorType.getShape().take_front(majorRank),
                        vectorType.getElementType());
    minorVectorType =
        VectorType::get(vectorType.getShape().take_back(minorRank),
                        vectorType.getElementType());
    /// Memref of minor vector type is used for individual transfers.
    memRefMinorVectorType =
        MemRefType::get(majorVectorType.getShape(), minorVectorType, {},
                        xferOp.getMemRefType().getMemorySpace());
  }

  LogicalResult doReplace();

private:
  /// Creates the loop nest on the "major" dimensions and calls the
  /// `loopBodyBuilder` lambda in the context of the loop nest.
  template <typename Lambda>
  void emitLoops(Lambda loopBodyBuilder);

  /// Operate within the body of `emitLoops` to:
  ///   1. Compute the indexings `majorIvs + majorOffsets`.
  ///   2. Compute a boolean that determines whether the first `majorIvs.rank()`
  ///      dimensions `majorIvs + majorOffsets` are all within `memrefBounds`.
  ///   3. Create an IfOp conditioned on the boolean in step 2.
  ///   4. Call a `thenBlockBuilder` and an `elseBlockBuilder` to append
  ///      operations to the IfOp blocks as appropriate.
  template <typename LambdaThen, typename LambdaElse>
  void emitInBounds(ValueRange majorIvs, ValueRange majorOffsets,
                    MemRefBoundsCapture &memrefBounds,
                    LambdaThen thenBlockBuilder, LambdaElse elseBlockBuilder);

  /// Common state to lower vector transfer ops.
  PatternRewriter &rewriter;
  Location loc;
  std::unique_ptr<ScopedContext> scope;
  ConcreteOp xferOp;
  Operation *op;
  // A vector transfer copies data between:
  //   - memref<(leading_dims) x (major_dims) x (minor_dims) x type>
  //   - vector<(major_dims) x (minor_dims) x type>
  unsigned minorRank;         // for now always 1
  unsigned majorRank;         // vector rank - minorRank
  unsigned leadingRank;       // memref rank - vector rank
  VectorType vectorType;      // vector<(major_dims) x (minor_dims) x type>
  VectorType majorVectorType; // vector<(major_dims) x type>
  VectorType minorVectorType; // vector<(minor_dims) x type>
  MemRefType memRefMinorVectorType; // memref<vector<(minor_dims) x type>>
};

template <typename ConcreteOp>
template <typename Lambda>
void NDTransferOpHelper<ConcreteOp>::emitLoops(Lambda loopBodyBuilder) {
  /// Loop nest operates on the major dimensions
  MemRefBoundsCapture memrefBoundsCapture(xferOp.memref());
  VectorBoundsCapture vectorBoundsCapture(majorVectorType);
  auto majorLbs = vectorBoundsCapture.getLbs();
  auto majorUbs = vectorBoundsCapture.getUbs();
  auto majorSteps = vectorBoundsCapture.getSteps();
  SmallVector<Value, 8> majorIvs(vectorBoundsCapture.rank());
  AffineLoopNestBuilder(majorIvs, majorLbs, majorUbs, majorSteps)([&] {
    ValueRange indices(xferOp.indices());
    loopBodyBuilder(majorIvs, indices.take_front(leadingRank),
                    indices.drop_front(leadingRank).take_front(majorRank),
                    indices.take_back(minorRank), memrefBoundsCapture);
  });
}

template <typename ConcreteOp>
template <typename LambdaThen, typename LambdaElse>
void NDTransferOpHelper<ConcreteOp>::emitInBounds(
    ValueRange majorIvs, ValueRange majorOffsets,
    MemRefBoundsCapture &memrefBounds, LambdaThen thenBlockBuilder,
    LambdaElse elseBlockBuilder) {
  Value inBounds;
  SmallVector<Value, 4> majorIvsPlusOffsets;
  majorIvsPlusOffsets.reserve(majorIvs.size());
  unsigned idx = 0;
  for (auto it : llvm::zip(majorIvs, majorOffsets, memrefBounds.getUbs())) {
    Value iv = std::get<0>(it), off = std::get<1>(it), ub = std::get<2>(it);
    using namespace mlir::edsc::op;
    majorIvsPlusOffsets.push_back(iv + off);
    if (xferOp.isMaskedDim(leadingRank + idx)) {
      Value inBounds2 = majorIvsPlusOffsets.back() < ub;
      inBounds = (inBounds) ? (inBounds && inBounds2) : inBounds2;
    }
    ++idx;
  }

  if (inBounds) {
    auto ifOp = ScopedContext::getBuilderRef().create<scf::IfOp>(
        ScopedContext::getLocation(), TypeRange{}, inBounds,
        /*withElseRegion=*/std::is_same<ConcreteOp, TransferReadOp>());
    BlockBuilder(&ifOp.thenRegion().front(),
                 Append())([&] { thenBlockBuilder(majorIvsPlusOffsets); });
    if (std::is_same<ConcreteOp, TransferReadOp>())
      BlockBuilder(&ifOp.elseRegion().front(),
                   Append())([&] { elseBlockBuilder(majorIvsPlusOffsets); });
  } else {
    // Just build the body of the then block right here.
    thenBlockBuilder(majorIvsPlusOffsets);
  }
}

template <>
LogicalResult NDTransferOpHelper<TransferReadOp>::doReplace() {
  Value alloc = std_alloc(memRefMinorVectorType);

  emitLoops([&](ValueRange majorIvs, ValueRange leadingOffsets,
                ValueRange majorOffsets, ValueRange minorOffsets,
                MemRefBoundsCapture &memrefBounds) {
    // If in-bounds, index into memref and lower to 1-D transfer read.
    auto thenBlockBuilder = [&](ValueRange majorIvsPlusOffsets) {
      SmallVector<Value, 8> indexing;
      indexing.reserve(leadingRank + majorRank + minorRank);
      indexing.append(leadingOffsets.begin(), leadingOffsets.end());
      indexing.append(majorIvsPlusOffsets.begin(), majorIvsPlusOffsets.end());
      indexing.append(minorOffsets.begin(), minorOffsets.end());

      Value memref = xferOp.memref();
      auto map = TransferReadOp::getTransferMinorIdentityMap(
          xferOp.getMemRefType(), minorVectorType);
      ArrayAttr masked;
      if (xferOp.isMaskedDim(xferOp.getVectorType().getRank() - 1)) {
        OpBuilder &b = ScopedContext::getBuilderRef();
        masked = b.getBoolArrayAttr({true});
      }
      auto loaded1D = vector_transfer_read(minorVectorType, memref, indexing,
                                           AffineMapAttr::get(map),
                                           xferOp.padding(), masked);
      // Store the 1-D vector.
      std_store(loaded1D, alloc, majorIvs);
    };
    // If out-of-bounds, just store a splatted vector.
    auto elseBlockBuilder = [&](ValueRange majorIvsPlusOffsets) {
      auto vector = std_splat(minorVectorType, xferOp.padding());
      std_store(vector, alloc, majorIvs);
    };
    emitInBounds(majorIvs, majorOffsets, memrefBounds, thenBlockBuilder,
                 elseBlockBuilder);
  });

  Value loaded =
      std_load(vector_type_cast(MemRefType::get({}, vectorType), alloc));
  rewriter.replaceOp(op, loaded);

  return success();
}

template <>
LogicalResult NDTransferOpHelper<TransferWriteOp>::doReplace() {
  Value alloc = std_alloc(memRefMinorVectorType);

  std_store(xferOp.vector(),
            vector_type_cast(MemRefType::get({}, vectorType), alloc));

  emitLoops([&](ValueRange majorIvs, ValueRange leadingOffsets,
                ValueRange majorOffsets, ValueRange minorOffsets,
                MemRefBoundsCapture &memrefBounds) {
    auto thenBlockBuilder = [&](ValueRange majorIvsPlusOffsets) {
      SmallVector<Value, 8> indexing;
      indexing.reserve(leadingRank + majorRank + minorRank);
      indexing.append(leadingOffsets.begin(), leadingOffsets.end());
      indexing.append(majorIvsPlusOffsets.begin(), majorIvsPlusOffsets.end());
      indexing.append(minorOffsets.begin(), minorOffsets.end());
      // Lower to 1-D vector_transfer_write and let recursion handle it.
      Value loaded1D = std_load(alloc, majorIvs);
      auto map = TransferWriteOp::getTransferMinorIdentityMap(
          xferOp.getMemRefType(), minorVectorType);
      ArrayAttr masked;
      if (xferOp.isMaskedDim(xferOp.getVectorType().getRank() - 1)) {
        OpBuilder &b = ScopedContext::getBuilderRef();
        masked = b.getBoolArrayAttr({true});
      }
      vector_transfer_write(loaded1D, xferOp.memref(), indexing,
                            AffineMapAttr::get(map), masked);
    };
    // Don't write anything when out of bounds.
    auto elseBlockBuilder = [&](ValueRange majorIvsPlusOffsets) {};
    emitInBounds(majorIvs, majorOffsets, memrefBounds, thenBlockBuilder,
                 elseBlockBuilder);
  });

  rewriter.eraseOp(op);

  return success();
}

/// Analyzes the `transfer` to find an access dimension along the fastest remote
/// MemRef dimension. If such a dimension with coalescing properties is found,
/// `pivs` and `vectorBoundsCapture` are swapped so that the invocation of
/// LoopNestBuilder captures it in the innermost loop.
template <typename TransferOpTy>
static int computeCoalescedIndex(TransferOpTy transfer) {
  // rank of the remote memory access, coalescing behavior occurs on the
  // innermost memory dimension.
  auto remoteRank = transfer.getMemRefType().getRank();
  // Iterate over the results expressions of the permutation map to determine
  // the loop order for creating pointwise copies between remote and local
  // memories.
  int coalescedIdx = -1;
  auto exprs = transfer.permutation_map().getResults();
  for (auto en : llvm::enumerate(exprs)) {
    auto dim = en.value().template dyn_cast<AffineDimExpr>();
    if (!dim) {
      continue;
    }
    auto memRefDim = dim.getPosition();
    if (memRefDim == remoteRank - 1) {
      // memRefDim has coalescing properties, it should be swapped in the last
      // position.
      assert(coalescedIdx == -1 && "Unexpected > 1 coalesced indices");
      coalescedIdx = en.index();
    }
  }
  return coalescedIdx;
}

/// Emits remote memory accesses that are clipped to the boundaries of the
/// MemRef.
template <typename TransferOpTy>
static SmallVector<Value, 8>
clip(TransferOpTy transfer, MemRefBoundsCapture &bounds, ArrayRef<Value> ivs) {
  using namespace mlir::edsc;

  Value zero(std_constant_index(0)), one(std_constant_index(1));
  SmallVector<Value, 8> memRefAccess(transfer.indices());
  SmallVector<Value, 8> clippedScalarAccessExprs(memRefAccess.size());
  // Indices accessing to remote memory are clipped and their expressions are
  // returned in clippedScalarAccessExprs.
  for (unsigned memRefDim = 0; memRefDim < clippedScalarAccessExprs.size();
       ++memRefDim) {
    // Linear search on a small number of entries.
    int loopIndex = -1;
    auto exprs = transfer.permutation_map().getResults();
    for (auto en : llvm::enumerate(exprs)) {
      auto expr = en.value();
      auto dim = expr.template dyn_cast<AffineDimExpr>();
      // Sanity check.
      assert(
          (dim || expr.template cast<AffineConstantExpr>().getValue() == 0) &&
          "Expected dim or 0 in permutationMap");
      if (dim && memRefDim == dim.getPosition()) {
        loopIndex = en.index();
        break;
      }
    }

    // We cannot distinguish atm between unrolled dimensions that implement
    // the "always full" tile abstraction and need clipping from the other
    // ones. So we conservatively clip everything.
    using namespace edsc::op;
    auto N = bounds.ub(memRefDim);
    auto i = memRefAccess[memRefDim];
    if (loopIndex < 0) {
      auto N_minus_1 = N - one;
      auto select_1 = std_select(i < N, i, N_minus_1);
      clippedScalarAccessExprs[memRefDim] =
          std_select(i < zero, zero, select_1);
    } else {
      auto ii = ivs[loopIndex];
      auto i_plus_ii = i + ii;
      auto N_minus_1 = N - one;
      auto select_1 = std_select(i_plus_ii < N, i_plus_ii, N_minus_1);
      clippedScalarAccessExprs[memRefDim] =
          std_select(i_plus_ii < zero, zero, select_1);
    }
  }

  return clippedScalarAccessExprs;
}

namespace {

/// Implements lowering of TransferReadOp and TransferWriteOp to a
/// proper abstraction for the hardware.
///
/// For now, we only emit a simple loop nest that performs clipped pointwise
/// copies from a remote to a locally allocated memory.
///
/// Consider the case:
///
/// ```mlir
///    // Read the slice `%A[%i0, %i1:%i1+256, %i2:%i2+32]` into
///    // vector<32x256xf32> and pad with %f0 to handle the boundary case:
///    %f0 = constant 0.0f : f32
///    scf.for %i0 = 0 to %0 {
///      scf.for %i1 = 0 to %1 step %c256 {
///        scf.for %i2 = 0 to %2 step %c32 {
///          %v = vector.transfer_read %A[%i0, %i1, %i2], %f0
///               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
///               memref<?x?x?xf32>, vector<32x256xf32>
///    }}}
/// ```
///
/// The rewriters construct loop and indices that access MemRef A in a pattern
/// resembling the following (while guaranteeing an always full-tile
/// abstraction):
///
/// ```mlir
///    scf.for %d2 = 0 to %c256 {
///      scf.for %d1 = 0 to %c32 {
///        %s = %A[%i0, %i1 + %d1, %i2 + %d2] : f32
///        %tmp[%d2, %d1] = %s
///      }
///    }
/// ```
///
/// In the current state, only a clipping transfer is implemented by `clip`,
/// which creates individual indexing expressions of the form:
///
/// ```mlir-dsc
///    auto condMax = i + ii < N;
///    auto max = std_select(condMax, i + ii, N - one)
///    auto cond = i + ii < zero;
///    std_select(cond, zero, max);
/// ```
///
/// In the future, clipping should not be the only way and instead we should
/// load vectors + mask them. Similarly on the write side, load/mask/store for
/// implementing RMW behavior.
///
/// Lowers TransferOp into a combination of:
///   1. local memory allocation;
///   2. perfect loop nest over:
///      a. scalar load/stores from local buffers (viewed as a scalar memref);
///      a. scalar store/load to original memref (with clipping).
///   3. vector_load/store
///   4. local memory deallocation.
/// Minor variations occur depending on whether a TransferReadOp or
/// a TransferWriteOp is rewritten.
template <typename TransferOpTy>
struct VectorTransferRewriter : public RewritePattern {
  explicit VectorTransferRewriter(MLIRContext *context)
      : RewritePattern(TransferOpTy::getOperationName(), 1, context) {}

  /// Used for staging the transfer in a local scalar buffer.
  MemRefType tmpMemRefType(TransferOpTy transfer) const {
    auto vectorType = transfer.getVectorType();
    return MemRefType::get(vectorType.getShape(), vectorType.getElementType(),
                           {}, 0);
  }

  /// Performs the rewrite.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

/// Lowers TransferReadOp into a combination of:
///   1. local memory allocation;
///   2. perfect loop nest over:
///      a. scalar load from local buffers (viewed as a scalar memref);
///      a. scalar store to original memref (with clipping).
///   3. vector_load from local buffer (viewed as a memref<1 x vector>);
///   4. local memory deallocation.
///
/// Lowers the data transfer part of a TransferReadOp while ensuring no
/// out-of-bounds accesses are possible. Out-of-bounds behavior is handled by
/// clipping. This means that a given value in memory can be read multiple
/// times and concurrently.
///
/// Important notes about clipping and "full-tiles only" abstraction:
/// =================================================================
/// When using clipping for dealing with boundary conditions, the same edge
/// value will appear multiple times (a.k.a edge padding). This is fine if the
/// subsequent vector operations are all data-parallel but **is generally
/// incorrect** in the presence of reductions or extract operations.
///
/// More generally, clipping is a scalar abstraction that is expected to work
/// fine as a baseline for CPUs and GPUs but not for vector_load and DMAs.
/// To deal with real vector_load and DMAs, a "padded allocation + view"
/// abstraction with the ability to read out-of-memref-bounds (but still within
/// the allocated region) is necessary.
///
/// Whether using scalar loops or vector_load/DMAs to perform the transfer,
/// junk values will be materialized in the vectors and generally need to be
/// filtered out and replaced by the "neutral element". This neutral element is
/// op-dependent so, in the future, we expect to create a vector filter and
/// apply it to a splatted constant vector with the proper neutral element at
/// each ssa-use. This filtering is not necessary for pure data-parallel
/// operations.
///
/// In the case of vector_store/DMAs, Read-Modify-Write will be required, which
/// also have concurrency implications. Note that by using clipped scalar stores
/// in the presence of data-parallel only operations, we generate code that
/// writes the same value multiple time on the edge locations.
///
/// TODO(ntv): implement alternatives to clipping.
/// TODO(ntv): support non-data-parallel operations.

/// Performs the rewrite.
template <>
LogicalResult VectorTransferRewriter<TransferReadOp>::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  using namespace mlir::edsc::op;

  TransferReadOp transfer = cast<TransferReadOp>(op);
  if (AffineMap::isMinorIdentity(transfer.permutation_map())) {
    // If > 1D, emit a bunch of loops around 1-D vector transfers.
    if (transfer.getVectorType().getRank() > 1)
      return NDTransferOpHelper<TransferReadOp>(rewriter, transfer).doReplace();
    // If 1-D this is now handled by the target-specific lowering.
    if (transfer.getVectorType().getRank() == 1)
      return failure();
  }

  // Conservative lowering to scalar load / stores.
  // 1. Setup all the captures.
  ScopedContext scope(rewriter, transfer.getLoc());
  StdIndexedValue remote(transfer.memref());
  MemRefBoundsCapture memRefBoundsCapture(transfer.memref());
  VectorBoundsCapture vectorBoundsCapture(transfer.vector());
  int coalescedIdx = computeCoalescedIndex(transfer);
  // Swap the vectorBoundsCapture which will reorder loop bounds.
  if (coalescedIdx >= 0)
    vectorBoundsCapture.swapRanges(vectorBoundsCapture.rank() - 1,
                                   coalescedIdx);

  auto lbs = vectorBoundsCapture.getLbs();
  auto ubs = vectorBoundsCapture.getUbs();
  SmallVector<Value, 8> steps;
  steps.reserve(vectorBoundsCapture.getSteps().size());
  for (auto step : vectorBoundsCapture.getSteps())
    steps.push_back(std_constant_index(step));

  // 2. Emit alloc-copy-load-dealloc.
  Value tmp = std_alloc(tmpMemRefType(transfer));
  StdIndexedValue local(tmp);
  Value vec = vector_type_cast(tmp);
  SmallVector<Value, 8> ivs(lbs.size());
  LoopNestBuilder(ivs, lbs, ubs, steps)([&] {
    // Swap the ivs which will reorder memory accesses.
    if (coalescedIdx >= 0)
      std::swap(ivs.back(), ivs[coalescedIdx]);
    // Computes clippedScalarAccessExprs in the loop nest scope (ivs exist).
    local(ivs) = remote(clip(transfer, memRefBoundsCapture, ivs));
  });
  Value vectorValue = std_load(vec);
  (std_dealloc(tmp)); // vexing parse

  // 3. Propagate.
  rewriter.replaceOp(op, vectorValue);
  return success();
}

/// Lowers TransferWriteOp into a combination of:
///   1. local memory allocation;
///   2. vector_store to local buffer (viewed as a memref<1 x vector>);
///   3. perfect loop nest over:
///      a. scalar load from local buffers (viewed as a scalar memref);
///      a. scalar store to original memref (with clipping).
///   4. local memory deallocation.
///
/// More specifically, lowers the data transfer part while ensuring no
/// out-of-bounds accesses are possible. Out-of-bounds behavior is handled by
/// clipping. This means that a given value in memory can be written to multiple
/// times and concurrently.
///
/// See `Important notes about clipping and full-tiles only abstraction` in the
/// description of `readClipped` above.
///
/// TODO(ntv): implement alternatives to clipping.
/// TODO(ntv): support non-data-parallel operations.
template <>
LogicalResult VectorTransferRewriter<TransferWriteOp>::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  using namespace edsc::op;

  TransferWriteOp transfer = cast<TransferWriteOp>(op);
  if (AffineMap::isMinorIdentity(transfer.permutation_map())) {
    // If > 1D, emit a bunch of loops around 1-D vector transfers.
    if (transfer.getVectorType().getRank() > 1)
      return NDTransferOpHelper<TransferWriteOp>(rewriter, transfer)
          .doReplace();
    // If 1-D this is now handled by the target-specific lowering.
    if (transfer.getVectorType().getRank() == 1)
      return failure();
  }

  // 1. Setup all the captures.
  ScopedContext scope(rewriter, transfer.getLoc());
  StdIndexedValue remote(transfer.memref());
  MemRefBoundsCapture memRefBoundsCapture(transfer.memref());
  Value vectorValue(transfer.vector());
  VectorBoundsCapture vectorBoundsCapture(transfer.vector());
  int coalescedIdx = computeCoalescedIndex(transfer);
  // Swap the vectorBoundsCapture which will reorder loop bounds.
  if (coalescedIdx >= 0)
    vectorBoundsCapture.swapRanges(vectorBoundsCapture.rank() - 1,
                                   coalescedIdx);

  auto lbs = vectorBoundsCapture.getLbs();
  auto ubs = vectorBoundsCapture.getUbs();
  SmallVector<Value, 8> steps;
  steps.reserve(vectorBoundsCapture.getSteps().size());
  for (auto step : vectorBoundsCapture.getSteps())
    steps.push_back(std_constant_index(step));

  // 2. Emit alloc-store-copy-dealloc.
  Value tmp = std_alloc(tmpMemRefType(transfer));
  StdIndexedValue local(tmp);
  Value vec = vector_type_cast(tmp);
  std_store(vectorValue, vec);
  SmallVector<Value, 8> ivs(lbs.size());
  LoopNestBuilder(ivs, lbs, ubs, steps)([&] {
    // Swap the ivs which will reorder memory accesses.
    if (coalescedIdx >= 0)
      std::swap(ivs.back(), ivs[coalescedIdx]);
    // Computes clippedScalarAccessExprs in the loop nest scope (ivs exist).
    remote(clip(transfer, memRefBoundsCapture, ivs)) = local(ivs);
  });
  (std_dealloc(tmp)); // vexing parse...

  rewriter.eraseOp(op);
  return success();
}

} // namespace

void mlir::populateVectorToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<VectorTransferRewriter<vector::TransferReadOp>,
                  VectorTransferRewriter<vector::TransferWriteOp>>(context);
}
