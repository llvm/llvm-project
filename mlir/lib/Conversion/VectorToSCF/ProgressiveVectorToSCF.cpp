//===- ProgressiveVectorToSCF.h - Convert vector to SCF dialect -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector transfer operations to SCF.
//
//===----------------------------------------------------------------------===//

#include <type_traits>

#include "mlir/Conversion/VectorToSCF/ProgressiveVectorToSCF.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using vector::TransferReadOp;
using vector::TransferWriteOp;

namespace {

/// Attribute name used for labeling transfer ops during progressive lowering.
static const char kPassLabel[] = "__vector_to_scf_lowering__";

/// Lower to 1D transfer ops. Target-specific lowering will lower those.
static const int64_t kTargetRank = 1;

/// Given a MemRefType with VectorType element type, unpack one dimension from
/// the VectorType into the MemRefType.
///
/// E.g.: memref<9xvector<5x6xf32>> --> memref<9x5xvector<6xf32>>
static MemRefType unpackOneDim(MemRefType type) {
  auto vectorType = type.getElementType().dyn_cast<VectorType>();
  auto memrefShape = type.getShape();
  SmallVector<int64_t, 8> newMemrefShape;
  newMemrefShape.append(memrefShape.begin(), memrefShape.end());
  newMemrefShape.push_back(vectorType.getDimSize(0));
  return MemRefType::get(newMemrefShape,
                         VectorType::get(vectorType.getShape().drop_front(),
                                         vectorType.getElementType()));
}

/// Helper data structure for data and mask buffers.
struct BufferAllocs {
  Value dataBuffer;
  Value maskBuffer;
};

/// Allocate temporary buffers for data (vector) and mask (if present).
/// TODO: Parallelism and threadlocal considerations.
template <typename OpTy>
static BufferAllocs allocBuffers(OpTy xferOp) {
  auto &b = ScopedContext::getBuilderRef();
  OpBuilder::InsertionGuard guard(b);
  Operation *scope =
      xferOp->template getParentWithTrait<OpTrait::AutomaticAllocationScope>();
  assert(scope && "Expected op to be inside automatic allocation scope");
  b.setInsertionPointToStart(&scope->getRegion(0).front());

  BufferAllocs result;
  auto bufferType = MemRefType::get({}, xferOp.getVectorType());
  result.dataBuffer = memref_alloca(bufferType).value;

  if (xferOp.mask()) {
    auto maskType = MemRefType::get({}, xferOp.mask().getType());
    Value maskBuffer = memref_alloca(maskType);
    memref_store(xferOp.mask(), maskBuffer);
    result.maskBuffer = memref_load(maskBuffer);
  }

  return result;
}

/// Given a vector transfer op, calculate which dimension of the `source`
/// memref should be unpacked in the next application of TransferOpConversion.
/// A return value of None indicates a broadcast.
template <typename OpTy>
static Optional<int64_t> unpackedDim(OpTy xferOp) {
  auto map = xferOp.permutation_map();
  if (auto expr = map.getResult(0).template dyn_cast<AffineDimExpr>()) {
    return expr.getPosition();
  }
  assert(xferOp.isBroadcastDim(0) &&
         "Expected AffineDimExpr or AffineConstantExpr");
  return None;
}

/// Compute the permutation map for the new (N-1)-D vector transfer op. This
/// map is identical to the current permutation map, but the first result is
/// omitted.
template <typename OpTy>
static AffineMap unpackedPermutationMap(OpTy xferOp, OpBuilder &builder) {
  auto map = xferOp.permutation_map();
  return AffineMap::get(
      map.getNumDims(), 0, map.getResults().drop_front(),
      builder.getContext());
}

/// Calculate the indices for the new vector transfer op.
///
/// E.g.: transfer_read %A[%a, %b, %c, %d] ... : vector<5x4x3xf32> ...
///       --> transfer_read %A[%a, %b + iv, %c, %d] ... vector<4x3f32>
///                                 ^^^^^^
///              `iv` is the iteration variable of the (new) surrounding loop.
template <typename OpTy>
static void getXferIndices(OpTy xferOp, Value iv,
                           SmallVector<Value, 8> &indices) {
  typename OpTy::Adaptor adaptor(xferOp);
  // Corresponding memref dim of the vector dim that is unpacked.
  auto dim = unpackedDim(xferOp);
  auto prevIndices = adaptor.indices();
  indices.append(prevIndices.begin(), prevIndices.end());

  bool isBroadcast = !dim.hasValue();
  if (!isBroadcast) {
    using edsc::op::operator+;
    indices[dim.getValue()] = adaptor.indices()[dim.getValue()] + iv;
  }
}

static void maybeYieldValue(
    bool hasRetVal, OpBuilder builder, Location loc, Value value) {
  if (hasRetVal) {
    builder.create<scf::YieldOp>(loc, value);
  } else {
    builder.create<scf::YieldOp>(loc);
  }
}

/// Generates a boolean Value that is true if the iv-th bit in xferOp's mask
/// is set to true. No such check is generated under following circumstances:
/// * xferOp does not have a mask.
/// * xferOp's mask is not 1D. (In case of (N>1)-D, a subvector of the mask is
///   computed and attached to the new transfer op in the pattern.)
/// * The to-be-unpacked dim of xferOp is a broadcast.
template <typename OpTy>
static Value generateMaskCheck(OpBuilder &builder, OpTy xferOp, Value iv) {
  if (!xferOp.mask())
    return Value();
  if (xferOp.getMaskType().getRank() != 1)
    return Value();
  if (xferOp.isBroadcastDim(0))
    return Value();

  auto ivI32 = std_index_cast(IntegerType::get(builder.getContext(), 32), iv);
  return vector_extract_element(xferOp.mask(), ivI32).value;
}

/// Helper function TransferOpConversion and TransferOp1dConversion.
/// Generate an in-bounds check if the transfer op may go out-of-bounds on the
/// specified dimension `dim` with the loop iteration variable `iv`.
/// E.g., when unpacking dimension 0 from:
/// ```
/// %vec = vector.transfer_read %A[%a, %b] %cst
///     : vector<5x4xf32>, memref<?x?xf32>
/// ```
/// An if check similar to this will be generated inside the loop:
/// ```
/// %d = memref.dim %A, %c0 : memref<?x?xf32>
/// if (%a + iv < %d) {
///   (in-bounds case)
/// } else {
///   (out-of-bounds case)
/// }
/// ```
///
/// If the transfer is 1D and has a mask, this function generates a more complex
/// check also accounts for potentially masked out elements.
///
/// This function variant returns the value returned by `inBoundsCase` or
/// `outOfBoundsCase`. The MLIR type of the return value must be specified in
/// `resultTypes`.
template <typename OpTy>
static Value generateInBoundsCheck(
    OpTy xferOp, Value iv, OpBuilder &builder, Optional<int64_t> dim,
    TypeRange resultTypes,
    function_ref<Value(OpBuilder &, Location)> inBoundsCase,
    function_ref<Value(OpBuilder &, Location)> outOfBoundsCase = nullptr) {
  bool hasRetVal = !resultTypes.empty();
  Value cond; // Condition to be built...

  // Condition check 1: Access in-bounds?
  bool isBroadcast = !dim.hasValue();  // No in-bounds check for broadcasts.
  if (!xferOp.isDimInBounds(0) && !isBroadcast) {
    auto memrefDim =
        memref_dim(xferOp.source(), std_constant_index(dim.getValue()));
    using edsc::op::operator+;
    auto memrefIdx = xferOp.indices()[dim.getValue()] + iv;
    cond = std_cmpi_sgt(memrefDim.value, memrefIdx);
  }

  // Condition check 2: Masked in?
  if (auto maskCond = generateMaskCheck(builder, xferOp, iv)) {
    if (cond) {
      cond = builder.create<AndOp>(xferOp.getLoc(), cond, maskCond);
    } else {
      cond = maskCond;
    }
  }

  // If the condition is non-empty, generate an SCF::IfOp.
  if (cond) {
    auto check = builder.create<scf::IfOp>(
        xferOp.getLoc(), resultTypes, cond,
        /*thenBuilder=*/[&](OpBuilder &builder, Location loc) {
      maybeYieldValue(hasRetVal, builder, loc, inBoundsCase(builder, loc));
    }, /*elseBuilder=*/[&](OpBuilder &builder, Location loc) {
      if (outOfBoundsCase) {
        maybeYieldValue(hasRetVal, builder, loc, outOfBoundsCase(builder, loc));
      } else {
        builder.create<scf::YieldOp>(loc);
      }
    });

    return hasRetVal ? check.getResult(0) : Value();
  }

  // Condition is empty, no need for an SCF::IfOp.
  return inBoundsCase(builder, xferOp.getLoc());
}

/// In this function variant, `inBoundsCase` and `outOfBoundsCase` do not have
/// a return value. Consequently, this function does not have a return value.
template <typename OpTy>
static void generateInBoundsCheck(
    OpTy xferOp, Value iv, OpBuilder &builder, Optional<int64_t> dim,
    function_ref<void(OpBuilder &, Location)> inBoundsCase,
    function_ref<void(OpBuilder &, Location)> outOfBoundsCase = nullptr) {
  generateInBoundsCheck(
      xferOp, iv, builder, dim, /*resultTypes=*/TypeRange(),
      /*inBoundsCase=*/[&](OpBuilder &builder, Location loc) {
        inBoundsCase(builder, loc);
        return Value();
      },
      /*outOfBoundsCase=*/[&](OpBuilder &builder, Location loc) {
        if (outOfBoundsCase)
            outOfBoundsCase(builder, loc);
        return Value();
      });
}

/// Given an ArrayAttr, return a copy where the first element is dropped.
static ArrayAttr dropFirstElem(OpBuilder &builder, ArrayAttr attr) {
  if (!attr)
      return attr;
  return ArrayAttr::get(builder.getContext(), attr.getValue().drop_front());
}

/// Add the pass label to a vector transfer op if its rank is not the target
/// rank.
template <typename OpTy>
static void maybeApplyPassLabel(OpBuilder &builder, OpTy newXferOp) {
  if (newXferOp.getVectorType().getRank() > kTargetRank)
    newXferOp->setAttr(kPassLabel, builder.getUnitAttr());
}

/// Given a transfer op, find the memref from which the mask is loaded. This
/// is similar to Strategy<TransferWriteOp>::getBuffer.
template <typename OpTy>
static Value getMaskBuffer(OpTy xferOp) {
  assert(xferOp.mask() && "Expected that transfer op has mask");
  auto loadOp = xferOp.mask().template getDefiningOp<memref::LoadOp>();
  assert(loadOp && "Expected transfer op mask produced by LoadOp");
  return loadOp.getMemRef();
}

/// Codegen strategy, depending on the operation.
template <typename OpTy>
struct Strategy;

/// Code strategy for vector TransferReadOp.
template<>
struct Strategy<TransferReadOp> {
  /// Find the StoreOp that is used for writing the current TransferReadOp's
  /// result to the temporary buffer allocation.
  static memref::StoreOp getStoreOp(TransferReadOp xferOp) {
    assert(xferOp->hasOneUse() && "Expected exactly one use of TransferReadOp");
    auto storeOp = dyn_cast<memref::StoreOp>(
        (*xferOp->use_begin()).getOwner());
    assert(storeOp && "Expected TransferReadOp result used by StoreOp");
    return storeOp;
  }

  /// Find the temporary buffer allocation. All labeled TransferReadOps are
  /// used like this, where %buf is either the buffer allocation or a type cast
  /// of the buffer allocation:
  /// ```
  /// %vec = vector.transfer_read ... { __vector_to_scf_lowering__ } ...
  /// memref.store %vec, %buf[...] ...
  /// ```
  static Value getBuffer(TransferReadOp xferOp) {
    return getStoreOp(xferOp).getMemRef();
  }

  /// Retrieve the indices of the current StoreOp that stores into the buffer.
  static void getBufferIndices(TransferReadOp xferOp,
                               SmallVector<Value, 8> &indices) {
    auto storeOp = getStoreOp(xferOp);
    auto prevIndices = memref::StoreOpAdaptor(storeOp).indices();
    indices.append(prevIndices.begin(), prevIndices.end());
  }

  /// Rewrite the TransferReadOp, assuming that there are no out-of-bounds
  /// accesses on the to-be-unpacked dimension.
  ///
  /// 1. Generate a new (N-1)-d TransferReadOp using the loop iteration
  ///    variable `iv`.
  /// 2. Store the result into the (already `vector.type_cast`ed) buffer.
  ///
  /// E.g.:
  /// ```
  /// %vec = vector.transfer_read %A[%a+%i, %b, %c], %cst
  ///     : memref<?x?x?xf32>, vector<4x3xf32>
  /// memref.store %vec, %buf[%i] : memref<5xvector<4x3xf32>>
  /// ```
  /// Is rewritten to:
  /// ```
  /// %casted = vector.type_cast %buf
  ///     : memref<5xvector<4x3xf32>> to memref<5x4xvector<3xf32>>
  /// for %j = 0 to 4 {
  ///   %vec = vector.transfer_read %A[%a+%i, %b+%j, %c], %cst
  ///       : memref<?x?x?xf32>, vector<3xf32>
  ///   memref.store %vec, %casted[%i, %j] : memref<5x4xvector<3xf32>>
  /// }
  /// ```
  ///
  /// Note: The loop and type cast are generated in TransferOpConversion.
  ///       The original TransferReadOp and store op are deleted in `cleanup`.
  /// Note: The `mask` operand is set in TransferOpConversion.
  static TransferReadOp rewriteOp(OpBuilder &builder, TransferReadOp xferOp,
                                  Value buffer, Value iv) {
    SmallVector<Value, 8> storeIndices;
    getBufferIndices(xferOp, storeIndices);
    storeIndices.push_back(iv);

    SmallVector<Value, 8> xferIndices;
    getXferIndices(xferOp, iv, xferIndices);

    auto bufferType = buffer.getType().dyn_cast<ShapedType>();
    auto vecType = bufferType.getElementType().dyn_cast<VectorType>();
    auto inBoundsAttr = dropFirstElem(builder, xferOp.in_boundsAttr());
    auto newXfer = vector_transfer_read(
        vecType, xferOp.source(), xferIndices,
        AffineMapAttr::get(unpackedPermutationMap(xferOp, builder)),
        xferOp.padding(), Value(), inBoundsAttr).value;

    maybeApplyPassLabel(builder,
                        dyn_cast<TransferReadOp>(newXfer.getDefiningOp()));

    memref_store(newXfer, buffer, storeIndices);
    return newXfer.getDefiningOp<TransferReadOp>();
  }

  /// Handle out-of-bounds accesses on the to-be-unpacked dimension: Write
  /// padding value to the temporary buffer.
  static void handleOutOfBoundsDim(
      OpBuilder &/*builder*/, TransferReadOp xferOp, Value buffer,
      Value iv) {
    SmallVector<Value, 8> storeIndices;
    getBufferIndices(xferOp, storeIndices);
    storeIndices.push_back(iv);

    auto bufferType = buffer.getType().dyn_cast<ShapedType>();
    auto vecType = bufferType.getElementType().dyn_cast<VectorType>();
    auto vec = std_splat(vecType, xferOp.padding());
    memref_store(vec, buffer, storeIndices);
  }

  /// Cleanup after rewriting the op.
  static void cleanup(PatternRewriter &rewriter, TransferReadOp xferOp) {
    rewriter.eraseOp(getStoreOp(xferOp));
    rewriter.eraseOp(xferOp);
  }
};

/// Codegen strategy for vector TransferWriteOp.
template<>
struct Strategy<TransferWriteOp> {
  /// Find the temporary buffer allocation. All labeled TransferWriteOps are
  /// used like this, where %buf is either the buffer allocation or a type cast
  /// of the buffer allocation:
  /// ```
  /// %vec = memref.load %buf[...] ...
  /// vector.transfer_write %vec ... { __vector_to_scf_lowering__ } ...
  /// ```
  static Value getBuffer(TransferWriteOp xferOp) {
    auto loadOp = xferOp.vector().getDefiningOp<memref::LoadOp>();
    assert(loadOp && "Expected transfer op vector produced by LoadOp");
    return loadOp.getMemRef();
  }

  /// Retrieve the indices of the current LoadOp that loads from the buffer.
  static void getBufferIndices(TransferWriteOp xferOp,
                               SmallVector<Value, 8> &indices) {
    auto loadOp = xferOp.vector().getDefiningOp<memref::LoadOp>();
    auto prevIndices = memref::LoadOpAdaptor(loadOp).indices();
    indices.append(prevIndices.begin(), prevIndices.end());
  }

  /// Rewrite the TransferWriteOp, assuming that there are no out-of-bounds
  /// accesses on the to-be-unpacked dimension.
  ///
  /// 1. Load an (N-1)-d vector from the (already `vector.type_cast`ed) buffer,
  ///    using the loop iteration variable `iv`.
  /// 2. Generate a new (N-1)-d TransferWriteOp, writing the loaded vector back
  ///    to memory.
  ///
  /// Note: For more details, see comments on Strategy<TransferReadOp>.
  static TransferWriteOp rewriteOp(OpBuilder &builder, TransferWriteOp xferOp,
                                   Value buffer, Value iv) {
    SmallVector<Value, 8> loadIndices;
    getBufferIndices(xferOp, loadIndices);
    loadIndices.push_back(iv);

    SmallVector<Value, 8> xferIndices;
    getXferIndices(xferOp, iv, xferIndices);

    auto vec = memref_load(buffer, loadIndices);
    auto inBoundsAttr = dropFirstElem(builder, xferOp.in_boundsAttr());
    auto newXfer = vector_transfer_write(
        Type(), vec, xferOp.source(), xferIndices,
        AffineMapAttr::get(unpackedPermutationMap(xferOp, builder)),
        Value(), inBoundsAttr);

    maybeApplyPassLabel(builder, newXfer.op);

    return newXfer;
  }

  /// Handle out-of-bounds accesses on the to-be-unpacked dimension.
  static void handleOutOfBoundsDim(
      OpBuilder &builder, TransferWriteOp xferOp, Value buffer,
      Value iv) {}

  /// Cleanup after rewriting the op.
  static void cleanup(PatternRewriter &rewriter, TransferWriteOp xferOp) {
    rewriter.eraseOp(xferOp);
  }
};

template <typename OpTy>
LogicalResult checkPrepareXferOp(OpTy xferOp) {
  if (xferOp->hasAttr(kPassLabel))
      return failure();
  if (xferOp.getVectorType().getRank() <= kTargetRank)
      return failure();
  return success();
}

/// Prepare a TransferReadOp for progressive lowering.
///
/// 1. Allocate a temporary buffer.
/// 2. Label the TransferReadOp, marking it eligible for progressive lowering.
/// 3. Store the result of the TransferReadOp into the temporary buffer.
/// 4. Load the result from the temporary buffer and replace all uses of the
///    original TransferReadOp with this load.
///
/// E.g.:
/// ```
/// %vec = vector.transfer_read %A[%a, %b, %c], %cst
///     : vector<5x4xf32>, memref<?x?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %0 = memref.alloca() : memref<vector<5x4xf32>>
/// %1 = vector.transfer_read %A[%a, %b, %c], %cst
///     { __vector_to_scf_lowering__ } : vector<5x4xf32>, memref<?x?x?xf32>
/// memref.store %1, %0[] : memref<vector<5x4xf32>>
/// %vec = memref.load %0[] : memref<vector<5x4xf32>>
/// ```
///
/// Note: A second temporary buffer may be allocated for the `mask` operand.
struct PrepareTransferReadConversion
    : public OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (checkPrepareXferOp(xferOp).failed())
      return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());
    auto buffers = allocBuffers(xferOp);
    auto *newXfer = rewriter.clone(*xferOp.getOperation());
    newXfer->setAttr(kPassLabel, rewriter.getUnitAttr());
    if (xferOp.mask()) {
      dyn_cast<TransferReadOp>(newXfer).maskMutable().assign(
          buffers.maskBuffer);
    }

    memref_store(newXfer->getResult(0), buffers.dataBuffer);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(xferOp, buffers.dataBuffer);

    return success();
  }
};

/// Prepare a TransferWriteOp for progressive lowering.
///
/// 1. Allocate a temporary buffer.
/// 2. Store the vector into the buffer.
/// 3. Load the vector from the buffer again.
/// 4. Use the loaded vector as a TransferWriteOp operand and label the op,
///    marking it eligible for progressive lowering via TransferOpConversion.
///
/// E.g.:
/// ```
/// vector.transfer_write %vec, %A[%a, %b, %c]
///     : vector<5x4xf32>, memref<?x?x?xf32>
/// ```
/// is rewritten to:
/// ```
/// %0 = memref.alloca() : memref<vector<5x4xf32>>
/// memref.store %vec, %0[] : memref<vector<5x4xf32>>
/// %1 = memref.load %0[] : memref<vector<5x4xf32>>
/// vector.transfer_write %1, %A[%a, %b, %c] { __vector_to_scf_lowering__ }
///     : vector<5x4xf32>, memref<?x?x?xf32>
/// ```
///
/// Note: A second temporary buffer may be allocated for the `mask` operand.
struct PrepareTransferWriteConversion
    : public OpRewritePattern<TransferWriteOp> {
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (checkPrepareXferOp(xferOp).failed())
      return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());
    auto buffers = allocBuffers(xferOp);
    memref_store(xferOp.vector(), buffers.dataBuffer);
    auto loadedVec = memref_load(buffers.dataBuffer);
    rewriter.updateRootInPlace(xferOp, [&]() {
      xferOp.vectorMutable().assign(loadedVec);
      xferOp->setAttr(kPassLabel, rewriter.getUnitAttr());
    });

    if (xferOp.mask()) {
      rewriter.updateRootInPlace(
          xferOp, [&]() { xferOp.maskMutable().assign(buffers.maskBuffer); });
    }

    return success();
  }
};

/// Progressive lowering of vector transfer ops: Unpack one dimension.
///
/// 1. Unpack one dimension from the current buffer type and cast the buffer
///    to that new type. E.g.:
///    ```
///    %vec = memref.load %0[%1] : memref<5xvector<4x3xf32>>
///    vector.transfer_write %vec ...
///    ```
///    The following cast is generated:
///    ```
///    %casted = vector.type_cast %0
///        : memref<5xvector<4x3xf32>> to memref<5x4xvector<3xf32>>
///    ```
/// 2. Generate a for loop and rewrite the transfer op according to the
///    corresponding Strategy<OpTy>. If the to-be-unpacked dimension can be
///    out-of-bounds, generate an if-check and handle both cases separately.
/// 3. Clean up according to the corresponding Strategy<OpTy>.
template <typename OpTy>
struct TransferOpConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp->hasAttr(kPassLabel))
        return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());

    // Find and cast data buffer. How the buffer can be found depends on OpTy.
    auto dataBuffer = Strategy<OpTy>::getBuffer(xferOp);
    auto dataBufferType = dataBuffer.getType().template dyn_cast<MemRefType>();
    auto castedDataType = unpackOneDim(dataBufferType);
    auto castedDataBuffer = vector_type_cast(castedDataType, dataBuffer);

    // If the xferOp has a mask: Find and cast mask buffer.
    Value castedMaskBuffer;
    if (xferOp.mask()) {
      auto maskBuffer = getMaskBuffer(xferOp);
      auto maskBufferType =
          maskBuffer.getType().template dyn_cast<MemRefType>();
      if (xferOp.isBroadcastDim(0) || xferOp.getMaskType().getRank() == 1) {
        // Do not unpack a dimension of the mask, if:
        // * To-be-unpacked transfer op dimension is a broadcast.
        // * Mask is 1D, i.e., the mask cannot be further unpacked.
        //   (That means that all remaining dimensions of the transfer op must
        //   be broadcasted.)
        castedMaskBuffer = maskBuffer;
      } else {
        auto castedMaskType = unpackOneDim(maskBufferType);
        castedMaskBuffer = vector_type_cast(castedMaskType, maskBuffer);
      }
    }

    // Loop bounds and step.
    auto lb = std_constant_index(0).value;
    auto ub = std_constant_index(
                  castedDataType.getDimSize(castedDataType.getRank() - 1))
                  .value;
    auto step = std_constant_index(1).value;

    // Generate for loop.
    rewriter.create<scf::ForOp>(
        xferOp.getLoc(), lb, ub, step, ValueRange(),
        [&](OpBuilder &b, Location loc, Value iv,
            ValueRange /*loopState*/) {
      ScopedContext scope(b, loc);
      generateInBoundsCheck(
          xferOp, iv, b, unpackedDim(xferOp),
          /*inBoundsCase=*/
          [&](OpBuilder &b, Location /*loc*/) {
            // Create new transfer op.
            OpTy newXfer =
                Strategy<OpTy>::rewriteOp(b, xferOp, castedDataBuffer, iv);

            // If old transfer op has a mask: Set mask on new transfer op.
            // Special case: If the mask of the old transfer op is 1D and the
            //               unpacked dim is not a broadcast, no mask is needed
            //               on the new transfer op.
            if (xferOp.mask() && (xferOp.isBroadcastDim(0) ||
                                  xferOp.getMaskType().getRank() > 1)) {
              OpBuilder::InsertionGuard guard(b);
              b.setInsertionPoint(newXfer); // Insert load before newXfer.

              SmallVector<Value, 8> loadIndices;
              Strategy<OpTy>::getBufferIndices(xferOp, loadIndices);
              // In case of broadcast: Use same indices to load from memref as
              // before.
              if (!xferOp.isBroadcastDim(0))
                loadIndices.push_back(iv);

              auto mask = memref_load(castedMaskBuffer, loadIndices);
              rewriter.updateRootInPlace(
                  newXfer, [&]() { newXfer.maskMutable().assign(mask); });
            }
          },
          /*outOfBoundsCase=*/
          [&](OpBuilder &b, Location /*loc*/) {
            Strategy<OpTy>::handleOutOfBoundsDim(b, xferOp, castedDataBuffer,
                                                 iv);
          });
      b.create<scf::YieldOp>(loc);
    });

    Strategy<OpTy>::cleanup(rewriter, xferOp);
    return success();
  }
};

/// If the original transfer op has a mask, compute the mask of the new transfer
/// op (for the current iteration `i`) and assign it.
template <typename OpTy>
static void maybeAssignMask(OpBuilder &builder, OpTy xferOp, OpTy newXferOp,
                            int64_t i) {
  if (!xferOp.mask())
    return;

  if (xferOp.isBroadcastDim(0)) {
    // To-be-unpacked dimension is a broadcast, which does not have a
    // corresponding mask dimension. Mask attribute remains unchanged.
    newXferOp.maskMutable().assign(xferOp.mask());
    return;
  }

  if (xferOp.getMaskType().getRank() > 1) {
    // Unpack one dimension of the mask.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(newXferOp); // Insert load before newXfer.

    llvm::SmallVector<int64_t, 1> indices({i});
    auto newMask = vector_extract(xferOp.mask(), indices).value;
    newXferOp.maskMutable().assign(newMask);
  }

  // If we end up here: The mask of the old transfer op is 1D and the unpacked
  // dim is not a broadcast, so no mask is needed on the new transfer op.
  // `generateInBoundsCheck` will have evaluated the mask already.
}

/// Progressive lowering of vector TransferReadOp with unrolling: Unpack one
/// dimension. This is similar to TransferOpConversion<TransferReadOp>, but no
/// memref buffer is allocated and the SCF loop is fully unrolled.
///
/// ```
/// E.g.:
/// ```
/// %vec = vector.transfer_read %A[%a, %b, %c], %padding
///     : memref<?x?x?xf32>, vector<5x4xf32>
/// ```
/// is rewritten to IR such as (simplified):
/// ```
/// %v_init = splat %padding : vector<5x4xf32>
/// %tmp0 = vector.transfer_read %A[%a, %b, %c], %padding
///     : memref<?x?x?xf32>, vector<4xf32>
/// %v0 = vector.insert %tmp0, %v_init[0] : vector<4xf32> into vector<5x4xf32>
/// %tmp1 = vector.transfer_read %A[%a, %b + 1, %c], %padding
///     : memref<?x?x?xf32>, vector<4xf32>
/// %v1 = vector.insert %tmp1, %v0[1] : vector<4xf32> into vector<5x4xf32>
/// ...
/// %tmp4 = vector.transfer_read %A[%a, %b + 4, %c], %padding
///     : memref<?x?x?xf32>, vector<4xf32>
/// %vec = vector.insert %tmp1, %v3[4] : vector<4xf32> into vector<5x4xf32>
/// ```
///
/// Note: A pass label is attached to new TransferReadOps, so that subsequent
/// applications of this pattern do not create an additional %v_init vector.
struct UnrollTransferReadConversion : public OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  /// Find the result vector %v_init or create a new vector if this the first
  /// application of the pattern.
  Value getResultVector(TransferReadOp xferOp,
                        PatternRewriter &rewriter) const {
    if (xferOp->hasAttr(kPassLabel)) {
      return getInsertOp(xferOp).dest();
    }
    return std_splat(xferOp.getVectorType(), xferOp.padding()).value;
  }

  /// Assuming that this not the first application of the pattern, return the
  /// vector.insert op in which the result of this transfer op is used.
  vector::InsertOp getInsertOp(TransferReadOp xferOp) const {
    Operation *xferOpUser = *xferOp->getUsers().begin();
    return dyn_cast<vector::InsertOp>(xferOpUser);
  }

  /// Assuming that this not the first application of the pattern, return the
  /// indices of the vector.insert op in which the result of this transfer op
  /// is used.
  void getInsertionIndices(TransferReadOp xferOp,
                           SmallVector<int64_t, 8> &indices) const {
    if (xferOp->hasAttr(kPassLabel)) {
      llvm::for_each(getInsertOp(xferOp).position(), [&](Attribute attr) {
        indices.push_back(attr.dyn_cast<IntegerAttr>().getInt());
      });
    }
  }

  /// Rewrite the op: Unpack one dimension. Can handle masks, out-of-bounds
  /// accesses, and broadcasts and transposes in permutation maps.
  LogicalResult matchAndRewrite(TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (xferOp.getVectorType().getRank() <= kTargetRank)
      return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());
    auto vec = getResultVector(xferOp, rewriter);
    auto vecType = vec.getType().dyn_cast<VectorType>();
    auto xferVecType = xferOp.getVectorType();
    auto newXferVecType = VectorType::get(xferVecType.getShape().drop_front(),
                                          xferVecType.getElementType());
    int64_t dimSize = xferVecType.getShape()[0];

    // Generate fully unrolled loop of transfer ops.
    for (int64_t i = 0; i < dimSize; ++i) {
      Value iv = std_constant_index(i);

      vec = generateInBoundsCheck(
          xferOp, iv, rewriter, unpackedDim(xferOp), TypeRange(vecType),
          /*inBoundsCase=*/
          [&](OpBuilder &b, Location loc) {
            ScopedContext scope(b, loc);

            // Indices for the new transfer op.
            SmallVector<Value, 8> xferIndices;
            getXferIndices(xferOp, iv, xferIndices);

            // Indices for the new vector.insert op.
            SmallVector<int64_t, 8> insertionIndices;
            getInsertionIndices(xferOp, insertionIndices);
            insertionIndices.push_back(i);

            auto inBoundsAttr = dropFirstElem(b, xferOp.in_boundsAttr());
            auto newXferOpVal =
                vector_transfer_read(
                    newXferVecType, xferOp.source(), xferIndices,
                    AffineMapAttr::get(unpackedPermutationMap(xferOp, b)),
                    xferOp.padding(), Value(), inBoundsAttr)
                    .value;
            auto newXferOp =
                dyn_cast<TransferReadOp>(newXferOpVal.getDefiningOp());

            maybeAssignMask(b, xferOp, newXferOp, i);
            maybeApplyPassLabel(b, newXferOp);

            return vector_insert(newXferOp, vec, insertionIndices).value;
          },
          /*outOfBoundsCase=*/
          [&](OpBuilder &b, Location loc) {
            // Loop through original (unmodified) vector.
            return vec;
          });
    }

    if (xferOp->hasAttr(kPassLabel)) {
      rewriter.replaceOp(getInsertOp(xferOp), vec);
      rewriter.eraseOp(xferOp);
    } else {
      rewriter.replaceOp(xferOp, vec);
    }

    return success();
  }
};

/// Progressive lowering of vector TransferWriteOp with unrolling: Unpack one
/// dimension. This is similar to TransferOpConversion<TransferWriteOp>, but no
/// memref buffer is allocated and the SCF loop is fully unrolled.
///
/// ```
/// E.g.:
/// ```
/// vector.transfer_write %vec, %A[%a, %b, %c]
///     : vector<5x4xf32>, memref<?x?x?xf32>
/// ```
/// is rewritten to IR such as (simplified):
/// ```
/// %v0 = vector.extract %vec[0] : vector<5x4xf32>
/// vector.transfer_write %v0, %A[%a, %b, %c] : vector<4xf32>, memref<...>
/// %v1 = vector.extract %vec[1] : vector<5x4xf32>
/// vector.transfer_write %v1, %A[%a, %b + 1, %c] : vector<4xf32>, memref<...>
/// ...
/// %v4 = vector.extract %vec[4] : vector<5x4xf32>
/// vector.transfer_write %v4, %A[%a, %b + 4, %c] : vector<4xf32>, memref<...>
/// ```
///
/// Note: A pass label is attached to new TransferWriteOps, so that subsequent
/// applications of this pattern can read the indices of previously generated
/// vector.extract ops.
struct UnrollTransferWriteConversion
    : public OpRewritePattern<TransferWriteOp> {
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;

  /// If this is not the first application of the pattern, find the original
  /// vector %vec that is written by this transfer op. Otherwise, return the
  /// vector of this transfer op.
  Value getDataVector(TransferWriteOp xferOp) const {
    if (xferOp->hasAttr(kPassLabel))
      return getExtractOp(xferOp).vector();
    return xferOp.vector();
  }

  /// Assuming that this is not the first application of the pattern, find the
  /// vector.extract op whose result is written by this transfer op.
  vector::ExtractOp getExtractOp(TransferWriteOp xferOp) const {
    return dyn_cast<vector::ExtractOp>(xferOp.vector().getDefiningOp());
  }

  void getExtractionIndices(TransferWriteOp xferOp,
                            SmallVector<int64_t, 8> &indices) const {
    if (xferOp->hasAttr(kPassLabel)) {
      llvm::for_each(getExtractOp(xferOp).position(), [&](Attribute attr) {
        indices.push_back(attr.dyn_cast<IntegerAttr>().getInt());
      });
    }
  }

  /// Rewrite the op: Unpack one dimension. Can handle masks, out-of-bounds
  /// accesses, and broadcasts and transposes in permutation maps.
  LogicalResult matchAndRewrite(TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (xferOp.getVectorType().getRank() <= kTargetRank)
      return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());
    auto vec = getDataVector(xferOp);
    auto xferVecType = xferOp.getVectorType();
    int64_t dimSize = xferVecType.getShape()[0];

    // Generate fully unrolled loop of transfer ops.
    for (int64_t i = 0; i < dimSize; ++i) {
      Value iv = std_constant_index(i);

      generateInBoundsCheck(
          xferOp, iv, rewriter, unpackedDim(xferOp),
          /*inBoundsCase=*/[&](OpBuilder &b, Location loc) {
            ScopedContext scope(b, loc);

            // Indices for the new transfer op.
            SmallVector<Value, 8> xferIndices;
            getXferIndices(xferOp, iv, xferIndices);

            // Indices for the new vector.extract op.
            SmallVector<int64_t, 8> extractionIndices;
            getExtractionIndices(xferOp, extractionIndices);
            extractionIndices.push_back(i);

            auto extracted = vector_extract(vec, extractionIndices).value;
            auto inBoundsAttr = dropFirstElem(b, xferOp.in_boundsAttr());

            auto newXferOp =
                vector_transfer_write(
                    Type(), extracted, xferOp.source(), xferIndices,
                    AffineMapAttr::get(unpackedPermutationMap(xferOp, b)),
                    Value(), inBoundsAttr)
                    .op;

            maybeAssignMask(b, xferOp, newXferOp, i);
            maybeApplyPassLabel(b, newXferOp);
          });
    }

    rewriter.eraseOp(xferOp);
    return success();
  }
};

/// Compute the indices into the memref for the LoadOp/StoreOp generated as
/// part of TransferOp1dConversion. Return the memref dimension on which
/// the transfer is operating. A return value of None indicates a broadcast.
template <typename OpTy>
static Optional<int64_t> get1dMemrefIndices(
    OpTy xferOp, Value iv, SmallVector<Value, 8> &memrefIndices) {
  auto indices = xferOp.indices();
  auto map = xferOp.permutation_map();

  memrefIndices.append(indices.begin(), indices.end());
  assert(map.getNumResults() == 1 &&
         "Expected 1 permutation map result for 1D transfer");
  if (auto expr = map.getResult(0).template dyn_cast<AffineDimExpr>()) {
    auto dim = expr.getPosition();
    using edsc::op::operator+;
    memrefIndices[dim] = memrefIndices[dim] + iv;
    return dim;
  }

  assert(xferOp.isBroadcastDim(0) &&
         "Expected AffineDimExpr or AffineConstantExpr");
  return None;
}

/// Codegen strategy for TransferOp1dConversion, depending on the
/// operation.
template <typename OpTy>
struct Strategy1d;

/// Codegen strategy for TransferReadOp.
template <>
struct Strategy1d<TransferReadOp> {
  static void generateForLoopBody(
      OpBuilder &builder, Location loc, TransferReadOp xferOp, Value iv,
      ValueRange loopState) {
    SmallVector<Value, 8> indices;
    auto dim = get1dMemrefIndices(xferOp, iv, indices);
    auto ivI32 = std_index_cast(
        IntegerType::get(builder.getContext(), 32), iv);
    auto vec = loopState[0];

    // In case of out-of-bounds access, leave `vec` as is (was initialized with
    // padding value).
    auto nextVec = generateInBoundsCheck(
        xferOp, iv, builder, dim, TypeRange(xferOp.getVectorType()),
        /*inBoundsCase=*/[&](OpBuilder& /*b*/, Location loc) {
      auto val = memref_load(xferOp.source(), indices);
      return vector_insert_element(val, vec, ivI32.value).value;
    }, /*outOfBoundsCase=*/[&](OpBuilder& /*b*/, Location loc) {
      return vec;
    });
    builder.create<scf::YieldOp>(loc, nextVec);
  }

  static Value initialLoopState(TransferReadOp xferOp) {
    // Inititalize vector with padding value.
    return std_splat(xferOp.getVectorType(), xferOp.padding()).value;
  }
};

/// Codegen strategy for TransferWriteOp.
template <>
struct Strategy1d<TransferWriteOp> {
  static void generateForLoopBody(
      OpBuilder &builder, Location loc, TransferWriteOp xferOp, Value iv,
      ValueRange /*loopState*/) {
    SmallVector<Value, 8> indices;
    auto dim = get1dMemrefIndices(xferOp, iv, indices);
    auto ivI32 = std_index_cast(
        IntegerType::get(builder.getContext(), 32), iv);

    // Nothing to do in case of out-of-bounds access.
    generateInBoundsCheck(
        xferOp, iv, builder, dim,
        /*inBoundsCase=*/[&](OpBuilder& /*b*/, Location loc) {
      auto val = vector_extract_element(xferOp.vector(), ivI32.value);
      memref_store(val, xferOp.source(), indices);
    });
    builder.create<scf::YieldOp>(loc);
  }

  static Value initialLoopState(TransferWriteOp xferOp) {
    return Value();
  }
};

/// Return true if the last dimension of the MemRefType has unit stride.
static bool isLastMemrefDimUnitStride(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  return succeeded(successStrides) && strides.back() == 1;
}

/// Lower a 1D vector transfer op to SCF using scalar loads/stores. This is
/// necessary in cases where a 1D vector transfer op cannot be lowered into
/// vector load/stores due to non-unit strides or broadcasts:
///
/// * Transfer dimension is not the last memref dimension
/// * Transfer dimension is a broadcast (i.e., scalar load + broadcast)
/// * Memref has a layout map with non-unit stride on the last dimension
///
/// This pattern generates IR as follows:
///
/// 1. Generate a for loop iterating over each vector element.
/// 2. Inside the loop, generate a InsertElementOp or ExtractElementOp,
///    depending on OpTy.
///
/// TODO: In some cases (no masking, etc.), LLVM::MatrixColumnMajorLoadOp
///       can be generated instead of TransferOp1dConversion. Add such a pattern
///       to ConvertVectorToLLVM.
///
/// E.g.:
/// ```
/// vector.transfer_write %vec, %A[%a, %b]
///    {permutation_map = affine_map<(d0, d1) -> (d0)>, in_bounds = [true]}
///    : vector<9xf32>, memref<?x?xf32>
/// ```
/// Is rewritten to approximately the following pseudo-IR:
/// ```
/// for i = 0 to 9 {
///   %t = vector.extractelement %vec[i] : vector<9xf32>
///   memref.store %t, %arg0[%a + i, %b] : memref<?x?xf32>
/// }
/// ```
template <typename OpTy>
struct TransferOp1dConversion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy xferOp,
                                PatternRewriter &rewriter) const override {
    ScopedContext scope(rewriter, xferOp.getLoc());
    auto map = xferOp.permutation_map();
    auto memRefType = xferOp.getShapedType().template dyn_cast<MemRefType>();

    if (!memRefType)
      return failure();
    if (xferOp.getVectorType().getRank() != 1)
      return failure();
    if (map.isMinorIdentity() && isLastMemrefDimUnitStride(memRefType))
      return failure(); // Handled by ConvertVectorToLLVM

    // Loop bounds, step, state...
    auto vecType = xferOp.getVectorType();
    auto lb = std_constant_index(0);
    auto ub = std_constant_index(vecType.getDimSize(0));
    auto step = std_constant_index(1);
    auto loopState = Strategy1d<OpTy>::initialLoopState(xferOp);

    // Generate for loop.
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        xferOp, lb, ub, step, loopState ? ValueRange(loopState) : ValueRange(),
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange loopState) {
      ScopedContext nestedScope(builder, loc);
      Strategy1d<OpTy>::generateForLoopBody(
          builder, loc, xferOp, iv, loopState);
    });

    return success();
  }
};

}  // namespace

namespace mlir {

void populateProgressiveVectorToSCFConversionPatterns(
    RewritePatternSet &patterns,
    const ProgressiveVectorTransferToSCFOptions &options) {
  if (options.unroll) {
    patterns.add<UnrollTransferReadConversion, UnrollTransferWriteConversion>(
        patterns.getContext());
  } else {
    patterns.add<PrepareTransferReadConversion, PrepareTransferWriteConversion,
                 TransferOpConversion<TransferReadOp>,
                 TransferOpConversion<TransferWriteOp>>(patterns.getContext());
  }

  if (kTargetRank == 1) {
    patterns.add<TransferOp1dConversion<TransferReadOp>,
                 TransferOp1dConversion<TransferWriteOp>>(
        patterns.getContext());
  }
}

struct ConvertProgressiveVectorToSCFPass
    : public ConvertVectorToSCFBase<ConvertProgressiveVectorToSCFPass> {
  ConvertProgressiveVectorToSCFPass(
      const ProgressiveVectorTransferToSCFOptions &opt)
      : options(opt) {}

  void runOnFunction() override {
    RewritePatternSet patterns(getFunction().getContext());
    populateProgressiveVectorToSCFConversionPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }

  ProgressiveVectorTransferToSCFOptions options;
};

}  // namespace mlir

std::unique_ptr<Pass> mlir::createProgressiveConvertVectorToSCFPass(
    const ProgressiveVectorTransferToSCFOptions &options) {
  return std::make_unique<ConvertProgressiveVectorToSCFPass>(options);
}
