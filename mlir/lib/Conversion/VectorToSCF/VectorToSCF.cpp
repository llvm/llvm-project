//===- VectorToSCF.cpp - Convert vector to SCF dialect ----------*- C++ -*-===//
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
#include <optional>

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTVECTORTOSCF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using vector::TransferReadOp;
using vector::TransferWriteOp;

namespace {

/// Attribute name used for labeling transfer ops during progressive lowering.
static const char kPassLabel[] = "__vector_to_scf_lowering__";

/// Patterns that inherit from this struct have access to
/// VectorTransferToSCFOptions.
template <typename OpTy>
struct VectorToSCFPattern : public OpRewritePattern<OpTy> {
  explicit VectorToSCFPattern(MLIRContext *context,
                              VectorTransferToSCFOptions opt)
      : OpRewritePattern<OpTy>(context), options(opt) {}

  VectorTransferToSCFOptions options;
};

/// Given a vector transfer op, calculate which dimension of the `source`
/// memref should be unpacked in the next application of TransferOpConversion.
/// A return value of std::nullopt indicates a broadcast.
template <typename OpTy>
static std::optional<int64_t> unpackedDim(OpTy xferOp) {
  // TODO: support 0-d corner case.
  assert(xferOp.getTransferRank() > 0 && "unexpected 0-d transfer");
  auto map = xferOp.getPermutationMap();
  if (auto expr = map.getResult(0).template dyn_cast<AffineDimExpr>()) {
    return expr.getPosition();
  }
  assert(xferOp.isBroadcastDim(0) &&
         "Expected AffineDimExpr or AffineConstantExpr");
  return std::nullopt;
}

/// Compute the permutation map for the new (N-1)-D vector transfer op. This
/// map is identical to the current permutation map, but the first result is
/// omitted.
template <typename OpTy>
static AffineMap unpackedPermutationMap(OpBuilder &b, OpTy xferOp) {
  // TODO: support 0-d corner case.
  assert(xferOp.getTransferRank() > 0 && "unexpected 0-d transfer");
  auto map = xferOp.getPermutationMap();
  return AffineMap::get(map.getNumDims(), 0, map.getResults().drop_front(),
                        b.getContext());
}

/// Calculate the indices for the new vector transfer op.
///
/// E.g.: transfer_read %A[%a, %b, %c, %d] ... : vector<5x4x3xf32> ...
///       --> transfer_read %A[%a, %b + iv, %c, %d] ... vector<4x3f32>
///                                 ^^^^^^
///              `iv` is the iteration variable of the (new) surrounding loop.
template <typename OpTy>
static void getXferIndices(OpBuilder &b, OpTy xferOp, Value iv,
                           SmallVector<Value, 8> &indices) {
  typename OpTy::Adaptor adaptor(xferOp);
  // Corresponding memref dim of the vector dim that is unpacked.
  auto dim = unpackedDim(xferOp);
  auto prevIndices = adaptor.getIndices();
  indices.append(prevIndices.begin(), prevIndices.end());

  Location loc = xferOp.getLoc();
  bool isBroadcast = !dim.has_value();
  if (!isBroadcast) {
    AffineExpr d0, d1;
    bindDims(xferOp.getContext(), d0, d1);
    Value offset = adaptor.getIndices()[*dim];
    indices[*dim] = makeComposedAffineApply(b, loc, d0 + d1, {offset, iv});
  }
}

static void maybeYieldValue(OpBuilder &b, Location loc, bool hasRetVal,
                            Value value) {
  if (hasRetVal) {
    assert(value && "Expected non-empty value");
    b.create<scf::YieldOp>(loc, value);
  } else {
    b.create<scf::YieldOp>(loc);
  }
}

/// Generates a boolean Value that is true if the iv-th bit in xferOp's mask
/// is set to true. No such check is generated under following circumstances:
/// * xferOp does not have a mask.
/// * xferOp's mask is not 1D. (In case of (N>1)-D, a subvector of the mask is
///   computed and attached to the new transfer op in the pattern.)
/// * The to-be-unpacked dim of xferOp is a broadcast.
template <typename OpTy>
static Value generateMaskCheck(OpBuilder &b, OpTy xferOp, Value iv) {
  if (!xferOp.getMask())
    return Value();
  if (xferOp.getMaskType().getRank() != 1)
    return Value();
  if (xferOp.isBroadcastDim(0))
    return Value();

  Location loc = xferOp.getLoc();
  return b.create<vector::ExtractElementOp>(loc, xferOp.getMask(), iv);
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
    OpBuilder &b, OpTy xferOp, Value iv, std::optional<int64_t> dim,
    TypeRange resultTypes,
    function_ref<Value(OpBuilder &, Location)> inBoundsCase,
    function_ref<Value(OpBuilder &, Location)> outOfBoundsCase = nullptr) {
  bool hasRetVal = !resultTypes.empty();
  Value cond; // Condition to be built...

  // Condition check 1: Access in-bounds?
  bool isBroadcast = !dim; // No in-bounds check for broadcasts.
  Location loc = xferOp.getLoc();
  ImplicitLocOpBuilder lb(xferOp.getLoc(), b);
  if (!xferOp.isDimInBounds(0) && !isBroadcast) {
    Value memrefDim =
        vector::createOrFoldDimOp(b, loc, xferOp.getSource(), *dim);
    AffineExpr d0, d1;
    bindDims(xferOp.getContext(), d0, d1);
    Value base = xferOp.getIndices()[*dim];
    Value memrefIdx = makeComposedAffineApply(b, loc, d0 + d1, {base, iv});
    cond = lb.create<arith::CmpIOp>(arith::CmpIPredicate::sgt, memrefDim,
                                    memrefIdx);
  }

  // Condition check 2: Masked in?
  if (auto maskCond = generateMaskCheck(b, xferOp, iv)) {
    if (cond)
      cond = lb.create<arith::AndIOp>(cond, maskCond);
    else
      cond = maskCond;
  }

  // If the condition is non-empty, generate an SCF::IfOp.
  if (cond) {
    auto check = lb.create<scf::IfOp>(
        cond,
        /*thenBuilder=*/
        [&](OpBuilder &b, Location loc) {
          maybeYieldValue(b, loc, hasRetVal, inBoundsCase(b, loc));
        },
        /*elseBuilder=*/
        [&](OpBuilder &b, Location loc) {
          if (outOfBoundsCase) {
            maybeYieldValue(b, loc, hasRetVal, outOfBoundsCase(b, loc));
          } else {
            b.create<scf::YieldOp>(loc);
          }
        });

    return hasRetVal ? check.getResult(0) : Value();
  }

  // Condition is empty, no need for an SCF::IfOp.
  return inBoundsCase(b, loc);
}

/// In this function variant, `inBoundsCase` and `outOfBoundsCase` do not have
/// a return value. Consequently, this function does not have a return value.
template <typename OpTy>
static void generateInBoundsCheck(
    OpBuilder &b, OpTy xferOp, Value iv, std::optional<int64_t> dim,
    function_ref<void(OpBuilder &, Location)> inBoundsCase,
    function_ref<void(OpBuilder &, Location)> outOfBoundsCase = nullptr) {
  generateInBoundsCheck(
      b, xferOp, iv, dim, /*resultTypes=*/TypeRange(),
      /*inBoundsCase=*/
      [&](OpBuilder &b, Location loc) {
        inBoundsCase(b, loc);
        return Value();
      },
      /*outOfBoundsCase=*/
      [&](OpBuilder &b, Location loc) {
        if (outOfBoundsCase)
          outOfBoundsCase(b, loc);
        return Value();
      });
}

/// Given an ArrayAttr, return a copy where the first element is dropped.
static ArrayAttr dropFirstElem(OpBuilder &b, ArrayAttr attr) {
  if (!attr)
    return attr;
  return ArrayAttr::get(b.getContext(), attr.getValue().drop_front());
}

/// Add the pass label to a vector transfer op if its rank is not the target
/// rank.
template <typename OpTy>
static void maybeApplyPassLabel(OpBuilder &b, OpTy newXferOp,
                                unsigned targetRank) {
  if (newXferOp.getVectorType().getRank() > targetRank)
    newXferOp->setAttr(kPassLabel, b.getUnitAttr());
}

/// Return true if this transfer op operates on a source tensor.
template <typename OpTy>
static bool isTensorOp(OpTy xferOp) {
  if (xferOp.getShapedType().template isa<RankedTensorType>()) {
    if (xferOp.getOperationName().equals(TransferWriteOp::getOperationName())) {
      // TransferWriteOps on tensors have a result.
      assert(xferOp->getNumResults() > 0);
    }
    return true;
  }
  return false;
}

namespace lowering_n_d {

/// Helper data structure for data and mask buffers.
struct BufferAllocs {
  Value dataBuffer;
  Value maskBuffer;
};

// TODO: Parallelism and threadlocal considerations with a ParallelScope trait.
static Operation *getAutomaticAllocationScope(Operation *op) {
  Operation *scope =
      op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
  assert(scope && "Expected op to be inside automatic allocation scope");
  return scope;
}

/// Allocate temporary buffers for data (vector) and mask (if present).
template <typename OpTy>
static BufferAllocs allocBuffers(OpBuilder &b, OpTy xferOp) {
  Location loc = xferOp.getLoc();
  OpBuilder::InsertionGuard guard(b);
  Operation *scope = getAutomaticAllocationScope(xferOp);
  assert(scope->getNumRegions() == 1 &&
         "AutomaticAllocationScope with >1 regions");
  b.setInsertionPointToStart(&scope->getRegion(0).front());

  BufferAllocs result;
  auto bufferType = MemRefType::get({}, xferOp.getVectorType());
  result.dataBuffer = b.create<memref::AllocaOp>(loc, bufferType);

  if (xferOp.getMask()) {
    auto maskType = MemRefType::get({}, xferOp.getMask().getType());
    auto maskBuffer = b.create<memref::AllocaOp>(loc, maskType);
    b.setInsertionPoint(xferOp);
    b.create<memref::StoreOp>(loc, xferOp.getMask(), maskBuffer);
    result.maskBuffer = b.create<memref::LoadOp>(loc, maskBuffer, ValueRange());
  }

  return result;
}

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

/// Given a transfer op, find the memref from which the mask is loaded. This
/// is similar to Strategy<TransferWriteOp>::getBuffer.
template <typename OpTy>
static Value getMaskBuffer(OpTy xferOp) {
  assert(xferOp.getMask() && "Expected that transfer op has mask");
  auto loadOp = xferOp.getMask().template getDefiningOp<memref::LoadOp>();
  assert(loadOp && "Expected transfer op mask produced by LoadOp");
  return loadOp.getMemRef();
}

/// Codegen strategy, depending on the operation.
template <typename OpTy>
struct Strategy;

/// Code strategy for vector TransferReadOp.
template <>
struct Strategy<TransferReadOp> {
  /// Find the StoreOp that is used for writing the current TransferReadOp's
  /// result to the temporary buffer allocation.
  static memref::StoreOp getStoreOp(TransferReadOp xferOp) {
    assert(xferOp->hasOneUse() && "Expected exactly one use of TransferReadOp");
    auto storeOp = dyn_cast<memref::StoreOp>((*xferOp->use_begin()).getOwner());
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
    auto prevIndices = memref::StoreOpAdaptor(storeOp).getIndices();
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
  static TransferReadOp rewriteOp(OpBuilder &b,
                                  VectorTransferToSCFOptions options,
                                  TransferReadOp xferOp, Value buffer, Value iv,
                                  ValueRange /*loopState*/) {
    SmallVector<Value, 8> storeIndices;
    getBufferIndices(xferOp, storeIndices);
    storeIndices.push_back(iv);

    SmallVector<Value, 8> xferIndices;
    getXferIndices(b, xferOp, iv, xferIndices);

    Location loc = xferOp.getLoc();
    auto bufferType = buffer.getType().dyn_cast<ShapedType>();
    auto vecType = bufferType.getElementType().dyn_cast<VectorType>();
    auto inBoundsAttr = dropFirstElem(b, xferOp.getInBoundsAttr());
    auto newXferOp = b.create<vector::TransferReadOp>(
        loc, vecType, xferOp.getSource(), xferIndices,
        AffineMapAttr::get(unpackedPermutationMap(b, xferOp)),
        xferOp.getPadding(), Value(), inBoundsAttr);

    maybeApplyPassLabel(b, newXferOp, options.targetRank);

    b.create<memref::StoreOp>(loc, newXferOp.getVector(), buffer, storeIndices);
    return newXferOp;
  }

  /// Handle out-of-bounds accesses on the to-be-unpacked dimension: Write
  /// padding value to the temporary buffer.
  static Value handleOutOfBoundsDim(OpBuilder &b, TransferReadOp xferOp,
                                    Value buffer, Value iv,
                                    ValueRange /*loopState*/) {
    SmallVector<Value, 8> storeIndices;
    getBufferIndices(xferOp, storeIndices);
    storeIndices.push_back(iv);

    Location loc = xferOp.getLoc();
    auto bufferType = buffer.getType().dyn_cast<ShapedType>();
    auto vecType = bufferType.getElementType().dyn_cast<VectorType>();
    auto vec = b.create<vector::SplatOp>(loc, vecType, xferOp.getPadding());
    b.create<memref::StoreOp>(loc, vec, buffer, storeIndices);

    return Value();
  }

  /// Cleanup after rewriting the op.
  static void cleanup(PatternRewriter &rewriter, TransferReadOp xferOp,
                      scf::ForOp /*forOp*/) {
    rewriter.eraseOp(getStoreOp(xferOp));
    rewriter.eraseOp(xferOp);
  }

  /// Return the initial loop state for the generated scf.for loop.
  static Value initialLoopState(TransferReadOp xferOp) { return Value(); }
};

/// Codegen strategy for vector TransferWriteOp.
template <>
struct Strategy<TransferWriteOp> {
  /// Find the temporary buffer allocation. All labeled TransferWriteOps are
  /// used like this, where %buf is either the buffer allocation or a type cast
  /// of the buffer allocation:
  /// ```
  /// %vec = memref.load %buf[...] ...
  /// vector.transfer_write %vec ... { __vector_to_scf_lowering__ } ...
  /// ```
  static Value getBuffer(TransferWriteOp xferOp) {
    auto loadOp = xferOp.getVector().getDefiningOp<memref::LoadOp>();
    assert(loadOp && "Expected transfer op vector produced by LoadOp");
    return loadOp.getMemRef();
  }

  /// Retrieve the indices of the current LoadOp that loads from the buffer.
  static void getBufferIndices(TransferWriteOp xferOp,
                               SmallVector<Value, 8> &indices) {
    auto loadOp = xferOp.getVector().getDefiningOp<memref::LoadOp>();
    auto prevIndices = memref::LoadOpAdaptor(loadOp).getIndices();
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
  static TransferWriteOp rewriteOp(OpBuilder &b,
                                   VectorTransferToSCFOptions options,
                                   TransferWriteOp xferOp, Value buffer,
                                   Value iv, ValueRange loopState) {
    SmallVector<Value, 8> loadIndices;
    getBufferIndices(xferOp, loadIndices);
    loadIndices.push_back(iv);

    SmallVector<Value, 8> xferIndices;
    getXferIndices(b, xferOp, iv, xferIndices);

    Location loc = xferOp.getLoc();
    auto vec = b.create<memref::LoadOp>(loc, buffer, loadIndices);
    auto inBoundsAttr = dropFirstElem(b, xferOp.getInBoundsAttr());
    auto source = loopState.empty() ? xferOp.getSource() : loopState[0];
    Type type = isTensorOp(xferOp) ? xferOp.getShapedType() : Type();
    auto newXferOp = b.create<vector::TransferWriteOp>(
        loc, type, vec, source, xferIndices,
        AffineMapAttr::get(unpackedPermutationMap(b, xferOp)), Value(),
        inBoundsAttr);

    maybeApplyPassLabel(b, newXferOp, options.targetRank);

    return newXferOp;
  }

  /// Handle out-of-bounds accesses on the to-be-unpacked dimension.
  static Value handleOutOfBoundsDim(OpBuilder &b, TransferWriteOp xferOp,
                                    Value buffer, Value iv,
                                    ValueRange loopState) {
    return isTensorOp(xferOp) ? loopState[0] : Value();
  }

  /// Cleanup after rewriting the op.
  static void cleanup(PatternRewriter &rewriter, TransferWriteOp xferOp,
                      scf::ForOp forOp) {
    if (isTensorOp(xferOp)) {
      assert(forOp->getNumResults() == 1 && "Expected one for loop result");
      rewriter.replaceOp(xferOp, forOp->getResult(0));
    } else {
      rewriter.eraseOp(xferOp);
    }
  }

  /// Return the initial loop state for the generated scf.for loop.
  static Value initialLoopState(TransferWriteOp xferOp) {
    return isTensorOp(xferOp) ? xferOp.getSource() : Value();
  }
};

template <typename OpTy>
LogicalResult checkPrepareXferOp(OpTy xferOp,
                                 VectorTransferToSCFOptions options) {
  if (xferOp->hasAttr(kPassLabel))
    return failure();
  if (xferOp.getVectorType().getRank() <= options.targetRank)
    return failure();
  if (isTensorOp(xferOp) && !options.lowerTensors)
    return failure();
  // Transfer ops that modify the element type are not supported atm.
  if (xferOp.getVectorType().getElementType() !=
      xferOp.getShapedType().getElementType())
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
    : public VectorToSCFPattern<TransferReadOp> {
  using VectorToSCFPattern<TransferReadOp>::VectorToSCFPattern;

  LogicalResult matchAndRewrite(TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (checkPrepareXferOp(xferOp, options).failed())
      return failure();

    auto buffers = allocBuffers(rewriter, xferOp);
    auto *newXfer = rewriter.clone(*xferOp.getOperation());
    newXfer->setAttr(kPassLabel, rewriter.getUnitAttr());
    if (xferOp.getMask()) {
      dyn_cast<TransferReadOp>(newXfer).getMaskMutable().assign(
          buffers.maskBuffer);
    }

    Location loc = xferOp.getLoc();
    rewriter.create<memref::StoreOp>(loc, newXfer->getResult(0),
                                     buffers.dataBuffer);
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
    : public VectorToSCFPattern<TransferWriteOp> {
  using VectorToSCFPattern<TransferWriteOp>::VectorToSCFPattern;

  LogicalResult matchAndRewrite(TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (checkPrepareXferOp(xferOp, options).failed())
      return failure();

    Location loc = xferOp.getLoc();
    auto buffers = allocBuffers(rewriter, xferOp);
    rewriter.create<memref::StoreOp>(loc, xferOp.getVector(),
                                     buffers.dataBuffer);
    auto loadedVec = rewriter.create<memref::LoadOp>(loc, buffers.dataBuffer);
    rewriter.updateRootInPlace(xferOp, [&]() {
      xferOp.getVectorMutable().assign(loadedVec);
      xferOp->setAttr(kPassLabel, rewriter.getUnitAttr());
    });

    if (xferOp.getMask()) {
      rewriter.updateRootInPlace(xferOp, [&]() {
        xferOp.getMaskMutable().assign(buffers.maskBuffer);
      });
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
///
/// Note: If the transfer op is a TransferWriteOp and operates on a tensor
/// source (as opposed to a memref source), then each iteration of the generated
/// scf.for loop yields the new tensor value. E.g.:
/// ```
/// %result = scf.for i = 0 to 5 {
///   %0 = memref.load %buffer[i] : memref<5xvector<4x3xf32>>
///   %1 = vector.transfer_write %0, %source[...]
///       : vector<4x3xf32>, tensor<5x4x3xf32>
///   scf.yield %1 : tensor<5x4x3xf32>
/// }
/// ```
template <typename OpTy>
struct TransferOpConversion : public VectorToSCFPattern<OpTy> {
  using VectorToSCFPattern<OpTy>::VectorToSCFPattern;

  void initialize() {
    // This pattern recursively unpacks one dimension at a time. The recursion
    // bounded as the rank is strictly decreasing.
    this->setHasBoundedRewriteRecursion();
  }

  LogicalResult matchAndRewrite(OpTy xferOp,
                                PatternRewriter &rewriter) const override {
    if (!xferOp->hasAttr(kPassLabel))
      return failure();

    // Find and cast data buffer. How the buffer can be found depends on OpTy.
    ImplicitLocOpBuilder locB(xferOp.getLoc(), rewriter);
    auto dataBuffer = Strategy<OpTy>::getBuffer(xferOp);
    auto dataBufferType = dataBuffer.getType().template dyn_cast<MemRefType>();
    auto castedDataType = unpackOneDim(dataBufferType);
    auto castedDataBuffer =
        locB.create<vector::TypeCastOp>(castedDataType, dataBuffer);

    // If the xferOp has a mask: Find and cast mask buffer.
    Value castedMaskBuffer;
    if (xferOp.getMask()) {
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
        castedMaskBuffer =
            locB.create<vector::TypeCastOp>(castedMaskType, maskBuffer);
      }
    }

    // Loop bounds and step.
    auto lb = locB.create<arith::ConstantIndexOp>(0);
    auto ub = locB.create<arith::ConstantIndexOp>(
        castedDataType.getDimSize(castedDataType.getRank() - 1));
    auto step = locB.create<arith::ConstantIndexOp>(1);
    // TransferWriteOps that operate on tensors return the modified tensor and
    // require a loop state.
    auto loopState = Strategy<OpTy>::initialLoopState(xferOp);

    // Generate for loop.
    auto result = locB.create<scf::ForOp>(
        lb, ub, step, loopState ? ValueRange(loopState) : ValueRange(),
        [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
          Type stateType = loopState.empty() ? Type() : loopState[0].getType();

          auto result = generateInBoundsCheck(
              b, xferOp, iv, unpackedDim(xferOp),
              stateType ? TypeRange(stateType) : TypeRange(),
              /*inBoundsCase=*/
              [&](OpBuilder &b, Location loc) {
                // Create new transfer op.
                OpTy newXfer = Strategy<OpTy>::rewriteOp(
                    b, this->options, xferOp, castedDataBuffer, iv, loopState);

                // If old transfer op has a mask: Set mask on new transfer op.
                // Special case: If the mask of the old transfer op is 1D and
                // the
                //               unpacked dim is not a broadcast, no mask is
                //               needed on the new transfer op.
                if (xferOp.getMask() && (xferOp.isBroadcastDim(0) ||
                                         xferOp.getMaskType().getRank() > 1)) {
                  OpBuilder::InsertionGuard guard(b);
                  b.setInsertionPoint(newXfer); // Insert load before newXfer.

                  SmallVector<Value, 8> loadIndices;
                  Strategy<OpTy>::getBufferIndices(xferOp, loadIndices);
                  // In case of broadcast: Use same indices to load from memref
                  // as before.
                  if (!xferOp.isBroadcastDim(0))
                    loadIndices.push_back(iv);

                  auto mask = b.create<memref::LoadOp>(loc, castedMaskBuffer,
                                                       loadIndices);
                  rewriter.updateRootInPlace(newXfer, [&]() {
                    newXfer.getMaskMutable().assign(mask);
                  });
                }

                return loopState.empty() ? Value() : newXfer->getResult(0);
              },
              /*outOfBoundsCase=*/
              [&](OpBuilder &b, Location /*loc*/) {
                return Strategy<OpTy>::handleOutOfBoundsDim(
                    b, xferOp, castedDataBuffer, iv, loopState);
              });

          maybeYieldValue(b, loc, !loopState.empty(), result);
        });

    Strategy<OpTy>::cleanup(rewriter, xferOp, result);
    return success();
  }
};

} // namespace lowering_n_d

namespace lowering_n_d_unrolled {

/// If the original transfer op has a mask, compute the mask of the new transfer
/// op (for the current iteration `i`) and assign it.
template <typename OpTy>
static void maybeAssignMask(OpBuilder &b, OpTy xferOp, OpTy newXferOp,
                            int64_t i) {
  if (!xferOp.getMask())
    return;

  if (xferOp.isBroadcastDim(0)) {
    // To-be-unpacked dimension is a broadcast, which does not have a
    // corresponding mask dimension. Mask attribute remains unchanged.
    newXferOp.getMaskMutable().assign(xferOp.getMask());
    return;
  }

  if (xferOp.getMaskType().getRank() > 1) {
    // Unpack one dimension of the mask.
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(newXferOp); // Insert load before newXfer.

    llvm::SmallVector<int64_t, 1> indices({i});
    Location loc = xferOp.getLoc();
    auto newMask = b.create<vector::ExtractOp>(loc, xferOp.getMask(), indices);
    newXferOp.getMaskMutable().assign(newMask);
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
/// Note: As an optimization, if the result of the original TransferReadOp
/// was directly inserted into another vector, no new %v_init vector is created.
/// Instead, the new TransferReadOp results are inserted into that vector.
struct UnrollTransferReadConversion
    : public VectorToSCFPattern<TransferReadOp> {
  using VectorToSCFPattern<TransferReadOp>::VectorToSCFPattern;

  void initialize() {
    // This pattern recursively unpacks one dimension at a time. The recursion
    // bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  /// Return the vector into which the newly created TransferReadOp results
  /// are inserted.
  Value getResultVector(TransferReadOp xferOp,
                        PatternRewriter &rewriter) const {
    if (auto insertOp = getInsertOp(xferOp))
      return insertOp.getDest();
    Location loc = xferOp.getLoc();
    return rewriter.create<vector::SplatOp>(loc, xferOp.getVectorType(),
                                            xferOp.getPadding());
  }

  /// If the result of the TransferReadOp has exactly one user, which is a
  /// vector::InsertOp, return that operation.
  vector::InsertOp getInsertOp(TransferReadOp xferOp) const {
    if (xferOp->hasOneUse()) {
      Operation *xferOpUser = *xferOp->getUsers().begin();
      if (auto insertOp = dyn_cast<vector::InsertOp>(xferOpUser))
        return insertOp;
    }

    return vector::InsertOp();
  }

  /// If the result of the TransferReadOp has exactly one user, which is a
  /// vector::InsertOp, return that operation's indices.
  void getInsertionIndices(TransferReadOp xferOp,
                           SmallVector<int64_t, 8> &indices) const {
    if (auto insertOp = getInsertOp(xferOp)) {
      for (Attribute attr : insertOp.getPosition())
        indices.push_back(attr.dyn_cast<IntegerAttr>().getInt());
    }
  }

  /// Rewrite the op: Unpack one dimension. Can handle masks, out-of-bounds
  /// accesses, and broadcasts and transposes in permutation maps.
  LogicalResult matchAndRewrite(TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (xferOp.getVectorType().getRank() <= options.targetRank)
      return failure();
    if (isTensorOp(xferOp) && !options.lowerTensors)
      return failure();
    // Transfer ops that modify the element type are not supported atm.
    if (xferOp.getVectorType().getElementType() !=
        xferOp.getShapedType().getElementType())
      return failure();

    auto insertOp = getInsertOp(xferOp);
    auto vec = getResultVector(xferOp, rewriter);
    auto vecType = vec.getType().dyn_cast<VectorType>();
    auto xferVecType = xferOp.getVectorType();
    auto newXferVecType = VectorType::get(xferVecType.getShape().drop_front(),
                                          xferVecType.getElementType());
    int64_t dimSize = xferVecType.getShape()[0];

    // Generate fully unrolled loop of transfer ops.
    Location loc = xferOp.getLoc();
    for (int64_t i = 0; i < dimSize; ++i) {
      Value iv = rewriter.create<arith::ConstantIndexOp>(loc, i);

      vec = generateInBoundsCheck(
          rewriter, xferOp, iv, unpackedDim(xferOp), TypeRange(vecType),
          /*inBoundsCase=*/
          [&](OpBuilder &b, Location loc) {
            // Indices for the new transfer op.
            SmallVector<Value, 8> xferIndices;
            getXferIndices(b, xferOp, iv, xferIndices);

            // Indices for the new vector.insert op.
            SmallVector<int64_t, 8> insertionIndices;
            getInsertionIndices(xferOp, insertionIndices);
            insertionIndices.push_back(i);

            auto inBoundsAttr = dropFirstElem(b, xferOp.getInBoundsAttr());
            auto newXferOp = b.create<vector::TransferReadOp>(
                loc, newXferVecType, xferOp.getSource(), xferIndices,
                AffineMapAttr::get(unpackedPermutationMap(b, xferOp)),
                xferOp.getPadding(), Value(), inBoundsAttr);
            maybeAssignMask(b, xferOp, newXferOp, i);
            return b.create<vector::InsertOp>(loc, newXferOp, vec,
                                              insertionIndices);
          },
          /*outOfBoundsCase=*/
          [&](OpBuilder &b, Location loc) {
            // Loop through original (unmodified) vector.
            return vec;
          });
    }

    if (insertOp) {
      // Rewrite single user of the old TransferReadOp, which was an InsertOp.
      rewriter.replaceOp(insertOp, vec);
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
/// Note: As an optimization, if the vector of the original TransferWriteOp
/// was directly extracted from another vector via an ExtractOp `a`, extract
/// the vectors for the newly generated TransferWriteOps from `a`'s input. By
/// doing so, `a` may become dead, and the number of ExtractOps generated during
/// recursive application of this pattern will be minimal.
struct UnrollTransferWriteConversion
    : public VectorToSCFPattern<TransferWriteOp> {
  using VectorToSCFPattern<TransferWriteOp>::VectorToSCFPattern;

  void initialize() {
    // This pattern recursively unpacks one dimension at a time. The recursion
    // bounded as the rank is strictly decreasing.
    setHasBoundedRewriteRecursion();
  }

  /// Return the vector from which newly generated ExtracOps will extract.
  Value getDataVector(TransferWriteOp xferOp) const {
    if (auto extractOp = getExtractOp(xferOp))
      return extractOp.getVector();
    return xferOp.getVector();
  }

  /// If the input of the given TransferWriteOp is an ExtractOp, return it.
  vector::ExtractOp getExtractOp(TransferWriteOp xferOp) const {
    if (auto *op = xferOp.getVector().getDefiningOp())
      return dyn_cast<vector::ExtractOp>(op);
    return vector::ExtractOp();
  }

  /// If the input of the given TransferWriteOp is an ExtractOp, return its
  /// indices.
  void getExtractionIndices(TransferWriteOp xferOp,
                            SmallVector<int64_t, 8> &indices) const {
    if (auto extractOp = getExtractOp(xferOp)) {
      for (Attribute attr : extractOp.getPosition())
        indices.push_back(attr.dyn_cast<IntegerAttr>().getInt());
    }
  }

  /// Rewrite the op: Unpack one dimension. Can handle masks, out-of-bounds
  /// accesses, and broadcasts and transposes in permutation maps.
  LogicalResult matchAndRewrite(TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (xferOp.getVectorType().getRank() <= options.targetRank)
      return failure();
    if (isTensorOp(xferOp) && !options.lowerTensors)
      return failure();
    // Transfer ops that modify the element type are not supported atm.
    if (xferOp.getVectorType().getElementType() !=
        xferOp.getShapedType().getElementType())
      return failure();

    auto vec = getDataVector(xferOp);
    auto xferVecType = xferOp.getVectorType();
    int64_t dimSize = xferVecType.getShape()[0];
    auto source = xferOp.getSource(); // memref or tensor to be written to.
    auto sourceType = isTensorOp(xferOp) ? xferOp.getShapedType() : Type();

    // Generate fully unrolled loop of transfer ops.
    Location loc = xferOp.getLoc();
    for (int64_t i = 0; i < dimSize; ++i) {
      Value iv = rewriter.create<arith::ConstantIndexOp>(loc, i);

      auto updatedSource = generateInBoundsCheck(
          rewriter, xferOp, iv, unpackedDim(xferOp),
          isTensorOp(xferOp) ? TypeRange(sourceType) : TypeRange(),
          /*inBoundsCase=*/
          [&](OpBuilder &b, Location loc) {
            // Indices for the new transfer op.
            SmallVector<Value, 8> xferIndices;
            getXferIndices(b, xferOp, iv, xferIndices);

            // Indices for the new vector.extract op.
            SmallVector<int64_t, 8> extractionIndices;
            getExtractionIndices(xferOp, extractionIndices);
            extractionIndices.push_back(i);

            auto extracted =
                b.create<vector::ExtractOp>(loc, vec, extractionIndices);
            auto inBoundsAttr = dropFirstElem(b, xferOp.getInBoundsAttr());
            auto newXferOp = b.create<vector::TransferWriteOp>(
                loc, sourceType, extracted, source, xferIndices,
                AffineMapAttr::get(unpackedPermutationMap(b, xferOp)), Value(),
                inBoundsAttr);

            maybeAssignMask(b, xferOp, newXferOp, i);

            return isTensorOp(xferOp) ? newXferOp->getResult(0) : Value();
          },
          /*outOfBoundsCase=*/
          [&](OpBuilder &b, Location loc) {
            return isTensorOp(xferOp) ? source : Value();
          });

      if (isTensorOp(xferOp))
        source = updatedSource;
    }

    if (isTensorOp(xferOp))
      rewriter.replaceOp(xferOp, source);
    else
      rewriter.eraseOp(xferOp);

    return success();
  }
};

} // namespace lowering_n_d_unrolled

namespace lowering_1_d {

/// Compute the indices into the memref for the LoadOp/StoreOp generated as
/// part of TransferOp1dConversion. Return the memref dimension on which
/// the transfer is operating. A return value of std::nullopt indicates a
/// broadcast.
template <typename OpTy>
static std::optional<int64_t>
get1dMemrefIndices(OpBuilder &b, OpTy xferOp, Value iv,
                   SmallVector<Value, 8> &memrefIndices) {
  auto indices = xferOp.getIndices();
  auto map = xferOp.getPermutationMap();
  assert(xferOp.getTransferRank() > 0 && "unexpected 0-d transfer");

  memrefIndices.append(indices.begin(), indices.end());
  assert(map.getNumResults() == 1 &&
         "Expected 1 permutation map result for 1D transfer");
  if (auto expr = map.getResult(0).template dyn_cast<AffineDimExpr>()) {
    Location loc = xferOp.getLoc();
    auto dim = expr.getPosition();
    AffineExpr d0, d1;
    bindDims(xferOp.getContext(), d0, d1);
    Value offset = memrefIndices[dim];
    memrefIndices[dim] = makeComposedAffineApply(b, loc, d0 + d1, {offset, iv});
    return dim;
  }

  assert(xferOp.isBroadcastDim(0) &&
         "Expected AffineDimExpr or AffineConstantExpr");
  return std::nullopt;
}

/// Codegen strategy for TransferOp1dConversion, depending on the
/// operation.
template <typename OpTy>
struct Strategy1d;

/// Codegen strategy for TransferReadOp.
template <>
struct Strategy1d<TransferReadOp> {
  static void generateForLoopBody(OpBuilder &b, Location loc,
                                  TransferReadOp xferOp, Value iv,
                                  ValueRange loopState) {
    SmallVector<Value, 8> indices;
    auto dim = get1dMemrefIndices(b, xferOp, iv, indices);
    auto vec = loopState[0];

    // In case of out-of-bounds access, leave `vec` as is (was initialized with
    // padding value).
    auto nextVec = generateInBoundsCheck(
        b, xferOp, iv, dim, TypeRange(xferOp.getVectorType()),
        /*inBoundsCase=*/
        [&](OpBuilder &b, Location loc) {
          Value val =
              b.create<memref::LoadOp>(loc, xferOp.getSource(), indices);
          return b.create<vector::InsertElementOp>(loc, val, vec, iv);
        },
        /*outOfBoundsCase=*/
        [&](OpBuilder & /*b*/, Location loc) { return vec; });
    b.create<scf::YieldOp>(loc, nextVec);
  }

  static Value initialLoopState(OpBuilder &b, TransferReadOp xferOp) {
    // Inititalize vector with padding value.
    Location loc = xferOp.getLoc();
    return b.create<vector::SplatOp>(loc, xferOp.getVectorType(),
                                     xferOp.getPadding());
  }
};

/// Codegen strategy for TransferWriteOp.
template <>
struct Strategy1d<TransferWriteOp> {
  static void generateForLoopBody(OpBuilder &b, Location loc,
                                  TransferWriteOp xferOp, Value iv,
                                  ValueRange /*loopState*/) {
    SmallVector<Value, 8> indices;
    auto dim = get1dMemrefIndices(b, xferOp, iv, indices);

    // Nothing to do in case of out-of-bounds access.
    generateInBoundsCheck(
        b, xferOp, iv, dim,
        /*inBoundsCase=*/[&](OpBuilder &b, Location loc) {
          auto val =
              b.create<vector::ExtractElementOp>(loc, xferOp.getVector(), iv);
          b.create<memref::StoreOp>(loc, val, xferOp.getSource(), indices);
        });
    b.create<scf::YieldOp>(loc);
  }

  static Value initialLoopState(OpBuilder &b, TransferWriteOp xferOp) {
    return Value();
  }
};

/// Return true if the last dimension of the MemRefType has unit stride.
static bool isLastMemrefDimUnitStride(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  auto successStrides = getStridesAndOffset(type, strides, offset);
  return succeeded(successStrides) && (strides.empty() || strides.back() == 1);
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
struct TransferOp1dConversion : public VectorToSCFPattern<OpTy> {
  using VectorToSCFPattern<OpTy>::VectorToSCFPattern;

  LogicalResult matchAndRewrite(OpTy xferOp,
                                PatternRewriter &rewriter) const override {
    // TODO: support 0-d corner case.
    if (xferOp.getTransferRank() == 0)
      return failure();
    auto map = xferOp.getPermutationMap();
    auto memRefType = xferOp.getShapedType().template dyn_cast<MemRefType>();

    if (!memRefType)
      return failure();
    if (xferOp.getVectorType().getRank() != 1)
      return failure();
    if (map.isMinorIdentity() && isLastMemrefDimUnitStride(memRefType))
      return failure(); // Handled by ConvertVectorToLLVM

    // Loop bounds, step, state...
    Location loc = xferOp.getLoc();
    auto vecType = xferOp.getVectorType();
    auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ub =
        rewriter.create<arith::ConstantIndexOp>(loc, vecType.getDimSize(0));
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto loopState = Strategy1d<OpTy>::initialLoopState(rewriter, xferOp);

    // Generate for loop.
    rewriter.replaceOpWithNewOp<scf::ForOp>(
        xferOp, lb, ub, step, loopState ? ValueRange(loopState) : ValueRange(),
        [&](OpBuilder &b, Location loc, Value iv, ValueRange loopState) {
          Strategy1d<OpTy>::generateForLoopBody(b, loc, xferOp, iv, loopState);
        });

    return success();
  }
};

} // namespace lowering_1_d
} // namespace

void mlir::populateVectorToSCFConversionPatterns(
    RewritePatternSet &patterns, const VectorTransferToSCFOptions &options) {
  if (options.unroll) {
    patterns.add<lowering_n_d_unrolled::UnrollTransferReadConversion,
                 lowering_n_d_unrolled::UnrollTransferWriteConversion>(
        patterns.getContext(), options);
  } else {
    patterns.add<lowering_n_d::PrepareTransferReadConversion,
                 lowering_n_d::PrepareTransferWriteConversion,
                 lowering_n_d::TransferOpConversion<TransferReadOp>,
                 lowering_n_d::TransferOpConversion<TransferWriteOp>>(
        patterns.getContext(), options);
  }

  if (options.targetRank == 1) {
    patterns.add<lowering_1_d::TransferOp1dConversion<TransferReadOp>,
                 lowering_1_d::TransferOp1dConversion<TransferWriteOp>>(
        patterns.getContext(), options);
  }
}

namespace {

struct ConvertVectorToSCFPass
    : public impl::ConvertVectorToSCFBase<ConvertVectorToSCFPass> {
  ConvertVectorToSCFPass() = default;
  ConvertVectorToSCFPass(const VectorTransferToSCFOptions &options) {
    this->fullUnroll = options.unroll;
    this->targetRank = options.targetRank;
    this->lowerTensors = options.lowerTensors;
  }

  void runOnOperation() override {
    VectorTransferToSCFOptions options;
    options.unroll = fullUnroll;
    options.targetRank = targetRank;
    options.lowerTensors = lowerTensors;

    // Lower permutation maps first.
    RewritePatternSet lowerTransferPatterns(&getContext());
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
        lowerTransferPatterns);
    (void)applyPatternsAndFoldGreedily(getOperation(),
                                       std::move(lowerTransferPatterns));

    RewritePatternSet patterns(&getContext());
    populateVectorToSCFConversionPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::createConvertVectorToSCFPass(const VectorTransferToSCFOptions &options) {
  return std::make_unique<ConvertVectorToSCFPass>(options);
}
