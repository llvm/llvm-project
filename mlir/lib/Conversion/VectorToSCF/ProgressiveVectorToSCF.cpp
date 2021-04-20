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

// TODO: Parallelism and threadlocal considerations.
static Value setAllocAtFunctionEntry(MemRefType type, Operation *op) {
  auto &b = ScopedContext::getBuilderRef();
  OpBuilder::InsertionGuard guard(b);
  Operation *scope =
      op->getParentWithTrait<OpTrait::AutomaticAllocationScope>();
  assert(scope && "Expected op to be inside automatic allocation scope");
  b.setInsertionPointToStart(&scope->getRegion(0).front());
  Value res = memref_alloca(type);
  return res;
}

/// Given a vector transfer op, calculate which dimension of the `source`
/// memref should be unpacked in the next application of TransferOpConversion.
template <typename OpTy>
static int64_t unpackedDim(OpTy xferOp) {
  return xferOp.getShapedType().getRank() - xferOp.getVectorType().getRank();
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
  using edsc::op::operator+;
  indices[dim] = adaptor.indices()[dim] + iv;
}

/// Generate an in-bounds check if the transfer op on the to-be-unpacked
/// dimension may go out-of-bounds.
template <typename OpTy>
static void generateInBoundsCheck(
    OpTy xferOp, Value iv, PatternRewriter &rewriter,
    function_ref<void(OpBuilder &, Location)> inBoundsCase,
    function_ref<void(OpBuilder &, Location)> outOfBoundsCase = nullptr) {
  // Corresponding memref dim of the vector dim that is unpacked.
  auto dim = unpackedDim(xferOp);

  if (!xferOp.isDimInBounds(0)) {
    auto memrefDim = memref_dim(xferOp.source(), std_constant_index(dim));
    using edsc::op::operator+;
    auto memrefIdx = xferOp.indices()[dim] + iv;
    auto cond = std_cmpi_sgt(memrefDim.value, memrefIdx);
    rewriter.create<scf::IfOp>(
        xferOp.getLoc(), cond,
        [&](OpBuilder &builder, Location loc) {
          inBoundsCase(builder, loc);
          builder.create<scf::YieldOp>(xferOp.getLoc());
        },
        [&](OpBuilder &builder, Location loc) {
          if (outOfBoundsCase)
            outOfBoundsCase(builder, loc);
          builder.create<scf::YieldOp>(xferOp.getLoc());
        });
  } else {
    // No runtime check needed if dim is guaranteed to be in-bounds.
    inBoundsCase(rewriter, xferOp.getLoc());
  }
}

/// Given an ArrayAttr, return a copy where the first element is dropped.
static ArrayAttr dropFirstElem(PatternRewriter &rewriter, ArrayAttr attr) {
  if (!attr)
    return attr;
  return ArrayAttr::get(rewriter.getContext(), attr.getValue().drop_front());
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

  /// Retrieve the indices of the current StoreOp.
  static void getStoreIndices(TransferReadOp xferOp,
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
  static void rewriteOp(PatternRewriter &rewriter, TransferReadOp xferOp,
                        Value buffer, Value iv) {
    SmallVector<Value, 8> storeIndices;
    getStoreIndices(xferOp, storeIndices);
    storeIndices.push_back(iv);

    SmallVector<Value, 8> xferIndices;
    getXferIndices(xferOp, iv, xferIndices);

    auto bufferType = buffer.getType().dyn_cast<ShapedType>();
    auto vecType = bufferType.getElementType().dyn_cast<VectorType>();
    auto map = getTransferMinorIdentityMap(xferOp.getShapedType(), vecType);
    auto inBoundsAttr = dropFirstElem(rewriter, xferOp.in_boundsAttr());
    auto newXfer = vector_transfer_read(vecType, xferOp.source(), xferIndices,
                                        AffineMapAttr::get(map),
                                        xferOp.padding(), Value(), inBoundsAttr)
                       .value;

    if (vecType.getRank() > kTargetRank)
      newXfer.getDefiningOp()->setAttr(kPassLabel, rewriter.getUnitAttr());

    memref_store(newXfer, buffer, storeIndices);
  }

  /// Handle out-of-bounds accesses on the to-be-unpacked dimension: Write
  /// padding value to the temporary buffer.
  static void handleOutOfBoundsDim(PatternRewriter &rewriter,
                                   TransferReadOp xferOp, Value buffer,
                                   Value iv) {
    SmallVector<Value, 8> storeIndices;
    getStoreIndices(xferOp, storeIndices);
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
    auto loadOp = xferOp.vector().getDefiningOp<memref::LoadOp>();
    assert(loadOp && "Expected transfer op vector produced by LoadOp");
    return loadOp.getMemRef();
  }

  /// Retrieve the indices of the current LoadOp.
  static void getLoadIndices(TransferWriteOp xferOp,
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
  static void rewriteOp(PatternRewriter &rewriter, TransferWriteOp xferOp,
                        Value buffer, Value iv) {
    SmallVector<Value, 8> loadIndices;
    getLoadIndices(xferOp, loadIndices);
    loadIndices.push_back(iv);

    SmallVector<Value, 8> xferIndices;
    getXferIndices(xferOp, iv, xferIndices);

    auto vec = memref_load(buffer, loadIndices);
    auto vecType = vec.value.getType().dyn_cast<VectorType>();
    auto map = getTransferMinorIdentityMap(xferOp.getShapedType(), vecType);
    auto inBoundsAttr = dropFirstElem(rewriter, xferOp.in_boundsAttr());
    auto newXfer =
        vector_transfer_write(Type(), vec, xferOp.source(), xferIndices,
                              AffineMapAttr::get(map), Value(), inBoundsAttr);

    if (vecType.getRank() > kTargetRank)
      newXfer.op->setAttr(kPassLabel, rewriter.getUnitAttr());
  }

  /// Handle out-of-bounds accesses on the to-be-unpacked dimension.
  static void handleOutOfBoundsDim(PatternRewriter &rewriter,
                                   TransferWriteOp xferOp, Value buffer,
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
  if (xferOp.mask())
    return failure();
  if (!xferOp.permutation_map().isMinorIdentity())
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
struct PrepareTransferReadConversion : public OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferReadOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (checkPrepareXferOp(xferOp).failed())
      return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());
    auto allocType = MemRefType::get({}, xferOp.getVectorType());
    auto buffer = setAllocAtFunctionEntry(allocType, xferOp);
    auto *newXfer = rewriter.clone(*xferOp.getOperation());
    newXfer->setAttr(kPassLabel, rewriter.getUnitAttr());
    memref_store(newXfer->getResult(0), buffer);
    rewriter.replaceOpWithNewOp<memref::LoadOp>(xferOp, buffer);

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
struct PrepareTransferWriteConversion
    : public OpRewritePattern<TransferWriteOp> {
  using OpRewritePattern<TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransferWriteOp xferOp,
                                PatternRewriter &rewriter) const override {
    if (checkPrepareXferOp(xferOp).failed())
      return failure();

    ScopedContext scope(rewriter, xferOp.getLoc());
    auto allocType = MemRefType::get({}, xferOp.getVectorType());
    auto buffer = setAllocAtFunctionEntry(allocType, xferOp);
    memref_store(xferOp.vector(), buffer);
    auto loadedVec = memref_load(buffer);

    rewriter.updateRootInPlace(xferOp, [&]() {
      xferOp.vectorMutable().assign(loadedVec);
      xferOp->setAttr(kPassLabel, rewriter.getUnitAttr());
    });

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
    // How the buffer can be found depends on OpTy.
    auto buffer = Strategy<OpTy>::getBuffer(xferOp);
    auto bufferType = buffer.getType().template dyn_cast<MemRefType>();
    auto castedType = unpackOneDim(bufferType);
    auto casted = vector_type_cast(castedType, buffer);

    auto lb = std_constant_index(0).value;
    auto ub =
        std_constant_index(castedType.getDimSize(castedType.getRank() - 1))
            .value;
    affineLoopBuilder(lb, ub, 1, [&](Value iv) {
      generateInBoundsCheck(
          xferOp, iv, rewriter,
          /*inBoundsCase=*/
          [&](OpBuilder & /*b*/, Location loc) {
            Strategy<OpTy>::rewriteOp(rewriter, xferOp, casted, iv);
          },
          /*outOfBoundsCase=*/
          [&](OpBuilder & /*b*/, Location loc) {
            Strategy<OpTy>::handleOutOfBoundsDim(rewriter, xferOp, casted, iv);
          });
    });

    Strategy<OpTy>::cleanup(rewriter, xferOp);
    return success();
  }
};

} // namespace

namespace mlir {

void populateProgressiveVectorToSCFConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<PrepareTransferReadConversion, PrepareTransferWriteConversion,
               TransferOpConversion<TransferReadOp>,
               TransferOpConversion<TransferWriteOp>>(patterns.getContext());
}

struct ConvertProgressiveVectorToSCFPass
    : public ConvertVectorToSCFBase<ConvertProgressiveVectorToSCFPass> {
  void runOnFunction() override {
    RewritePatternSet patterns(getFunction().getContext());
    populateProgressiveVectorToSCFConversionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

} // namespace mlir

std::unique_ptr<Pass> mlir::createProgressiveConvertVectorToSCFPass() {
  return std::make_unique<ConvertProgressiveVectorToSCFPass>();
}
