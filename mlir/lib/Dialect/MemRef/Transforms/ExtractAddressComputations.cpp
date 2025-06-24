//===- ExtractAddressCmoputations.cpp - Extract address computations  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This transformation pass rewrites loading/storing from/to a memref with
/// offsets into loading/storing from/to a subview and without any offset on
/// the instruction itself.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helper functions for the `load base[off0...]`
//  => `load (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getFailureOrSrcMemRef specs for LoadOp.
// \see LoadStoreLikeOpRewriter.
static FailureOr<Value> getLoadOpSrcMemRef(memref::LoadOp loadOp) {
  return loadOp.getMemRef();
}

// Matches rebuildOpFromAddressAndIndices specs for LoadOp.
// \see LoadStoreLikeOpRewriter.
static memref::LoadOp rebuildLoadOp(RewriterBase &rewriter,
                                    memref::LoadOp loadOp, Value srcMemRef,
                                    ArrayRef<Value> indices) {
  Location loc = loadOp.getLoc();
  return rewriter.create<memref::LoadOp>(loc, srcMemRef, indices,
                                         loadOp.getNontemporal());
}

// Matches getViewSizeForEachDim specs for LoadOp.
// \see LoadStoreLikeOpRewriter.
static SmallVector<OpFoldResult>
getLoadOpViewSizeForEachDim(RewriterBase &rewriter, memref::LoadOp loadOp) {
  MemRefType ldTy = loadOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  return SmallVector<OpFoldResult>(loadRank, rewriter.getIndexAttr(1));
}

//===----------------------------------------------------------------------===//
// Helper functions for the `store val, base[off0...]`
//  => `store val, (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getFailureOrSrcMemRef specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static FailureOr<Value> getStoreOpSrcMemRef(memref::StoreOp storeOp) {
  return storeOp.getMemRef();
}

// Matches rebuildOpFromAddressAndIndices specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static memref::StoreOp rebuildStoreOp(RewriterBase &rewriter,
                                      memref::StoreOp storeOp, Value srcMemRef,
                                      ArrayRef<Value> indices) {
  Location loc = storeOp.getLoc();
  return rewriter.create<memref::StoreOp>(loc, storeOp.getValueToStore(),
                                          srcMemRef, indices,
                                          storeOp.getNontemporal());
}

// Matches getViewSizeForEachDim specs for StoreOp.
// \see LoadStoreLikeOpRewriter.
static SmallVector<OpFoldResult>
getStoreOpViewSizeForEachDim(RewriterBase &rewriter, memref::StoreOp storeOp) {
  MemRefType ldTy = storeOp.getMemRefType();
  unsigned loadRank = ldTy.getRank();
  return SmallVector<OpFoldResult>(loadRank, rewriter.getIndexAttr(1));
}

//===----------------------------------------------------------------------===//
// Helper functions for the `ldmatrix base[off0...]`
//  => `ldmatrix (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getFailureOrSrcMemRef specs for LdMatrixOp.
// \see LoadStoreLikeOpRewriter.
static FailureOr<Value> getLdMatrixOpSrcMemRef(nvgpu::LdMatrixOp ldMatrixOp) {
  return ldMatrixOp.getSrcMemref();
}

// Matches rebuildOpFromAddressAndIndices specs for LdMatrixOp.
// \see LoadStoreLikeOpRewriter.
static nvgpu::LdMatrixOp rebuildLdMatrixOp(RewriterBase &rewriter,
                                           nvgpu::LdMatrixOp ldMatrixOp,
                                           Value srcMemRef,
                                           ArrayRef<Value> indices) {
  Location loc = ldMatrixOp.getLoc();
  return rewriter.create<nvgpu::LdMatrixOp>(
      loc, ldMatrixOp.getResult().getType(), srcMemRef, indices,
      ldMatrixOp.getTranspose(), ldMatrixOp.getNumTiles());
}

//===----------------------------------------------------------------------===//
// Helper functions for the `transfer_read base[off0...]`
//  => `transfer_read (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches getFailureOrSrcMemRef specs for TransferReadOp.
// \see LoadStoreLikeOpRewriter.
template <typename TransferLikeOp>
static FailureOr<Value>
getTransferLikeOpSrcMemRef(TransferLikeOp transferLikeOp) {
  Value src = transferLikeOp.getBase();
  if (isa<MemRefType>(src.getType()))
    return src;
  return failure();
}

// Matches rebuildOpFromAddressAndIndices specs for TransferReadOp.
// \see LoadStoreLikeOpRewriter.
static vector::TransferReadOp
rebuildTransferReadOp(RewriterBase &rewriter,
                      vector::TransferReadOp transferReadOp, Value srcMemRef,
                      ArrayRef<Value> indices) {
  Location loc = transferReadOp.getLoc();
  return rewriter.create<vector::TransferReadOp>(
      loc, transferReadOp.getResult().getType(), srcMemRef, indices,
      transferReadOp.getPermutationMap(), transferReadOp.getPadding(),
      transferReadOp.getMask(), transferReadOp.getInBoundsAttr());
}

//===----------------------------------------------------------------------===//
// Helper functions for the `transfer_write base[off0...]`
//  => `transfer_write (subview base[off0...])[0...]` pattern.
//===----------------------------------------------------------------------===//

// Matches rebuildOpFromAddressAndIndices specs for TransferWriteOp.
// \see LoadStoreLikeOpRewriter.
static vector::TransferWriteOp
rebuildTransferWriteOp(RewriterBase &rewriter,
                       vector::TransferWriteOp transferWriteOp, Value srcMemRef,
                       ArrayRef<Value> indices) {
  Location loc = transferWriteOp.getLoc();
  return rewriter.create<vector::TransferWriteOp>(
      loc, transferWriteOp.getValue(), srcMemRef, indices,
      transferWriteOp.getPermutationMapAttr(), transferWriteOp.getMask(),
      transferWriteOp.getInBoundsAttr());
}

//===----------------------------------------------------------------------===//
// Generic helper functions used as default implementation in
// LoadStoreLikeOpRewriter.
//===----------------------------------------------------------------------===//

/// Helper function to get the src memref.
/// It uses the already defined getFailureOrSrcMemRef but asserts
/// that the source is a memref.
template <typename LoadStoreLikeOp,
          FailureOr<Value> (*getFailureOrSrcMemRef)(LoadStoreLikeOp)>
static Value getSrcMemRef(LoadStoreLikeOp loadStoreLikeOp) {
  FailureOr<Value> failureOrSrcMemRef = getFailureOrSrcMemRef(loadStoreLikeOp);
  assert(!failed(failureOrSrcMemRef) && "Generic getSrcMemRef cannot be used");
  return *failureOrSrcMemRef;
}

/// Helper function to get the sizes of the resulting view.
/// This function gets the sizes of the source memref then substracts the
/// offsets used within \p loadStoreLikeOp. This gives the maximal (for
/// inbound) sizes for the view.
/// The source memref is retrieved using getSrcMemRef on \p loadStoreLikeOp.
template <typename LoadStoreLikeOp, Value (*getSrcMemRef)(LoadStoreLikeOp)>
static SmallVector<OpFoldResult>
getGenericOpViewSizeForEachDim(RewriterBase &rewriter,
                               LoadStoreLikeOp loadStoreLikeOp) {
  Location loc = loadStoreLikeOp.getLoc();
  auto extractStridedMetadataOp =
      rewriter.create<memref::ExtractStridedMetadataOp>(
          loc, getSrcMemRef(loadStoreLikeOp));
  SmallVector<OpFoldResult> srcSizes =
      extractStridedMetadataOp.getConstifiedMixedSizes();
  SmallVector<OpFoldResult> indices =
      getAsOpFoldResult(loadStoreLikeOp.getIndices());
  SmallVector<OpFoldResult> finalSizes;

  AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
  AffineExpr s1 = rewriter.getAffineSymbolExpr(1);

  for (auto [srcSize, indice] : llvm::zip(srcSizes, indices)) {
    finalSizes.push_back(affine::makeComposedFoldedAffineApply(
        rewriter, loc, s0 - s1, {srcSize, indice}));
  }
  return finalSizes;
}

/// Rewrite a store/load-like op so that all its indices are zeros.
/// E.g., %ld = memref.load %base[%off0]...[%offN]
/// =>
/// %new_base = subview %base[%off0,.., %offN][1,..,1][1,..,1]
/// %ld = memref.load %new_base[0,..,0] :
///    memref<1x..x1xTy, strided<[1,..,1], offset: ?>>
///
/// `getSrcMemRef` returns the source memref for the given load-like operation.
///
/// `getViewSizeForEachDim` returns the sizes of view that is going to feed
/// new operation. This must return one size per dimension of the view.
/// The sizes of the view needs to be at least as big as what is actually
/// going to be accessed. Use the provided `loadStoreOp` to get the right
/// sizes.
///
/// Using the given rewriter, `rebuildOpFromAddressAndIndices` creates a new
/// LoadStoreLikeOp that reads from srcMemRef[indices].
/// The returned operation will be used to replace loadStoreOp.
template <typename LoadStoreLikeOp,
          FailureOr<Value> (*getFailureOrSrcMemRef)(LoadStoreLikeOp),
          LoadStoreLikeOp (*rebuildOpFromAddressAndIndices)(
              RewriterBase & /*rewriter*/, LoadStoreLikeOp /*loadStoreOp*/,
              Value /*srcMemRef*/, ArrayRef<Value> /*indices*/),
          SmallVector<OpFoldResult> (*getViewSizeForEachDim)(
              RewriterBase & /*rewriter*/, LoadStoreLikeOp /*loadStoreOp*/) =
              getGenericOpViewSizeForEachDim<
                  LoadStoreLikeOp,
                  getSrcMemRef<LoadStoreLikeOp, getFailureOrSrcMemRef>>>
struct LoadStoreLikeOpRewriter : public OpRewritePattern<LoadStoreLikeOp> {
  using OpRewritePattern<LoadStoreLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadStoreLikeOp loadStoreLikeOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<Value> failureOrSrcMemRef =
        getFailureOrSrcMemRef(loadStoreLikeOp);
    if (failed(failureOrSrcMemRef))
      return rewriter.notifyMatchFailure(loadStoreLikeOp,
                                         "source is not a memref");
    Value srcMemRef = *failureOrSrcMemRef;
    auto ldStTy = cast<MemRefType>(srcMemRef.getType());
    unsigned loadStoreRank = ldStTy.getRank();
    // Don't waste compile time if there is nothing to rewrite.
    if (loadStoreRank == 0)
      return rewriter.notifyMatchFailure(loadStoreLikeOp,
                                         "0-D accesses don't need rewriting");

    // If our load already has only zeros as indices there is nothing
    // to do.
    SmallVector<OpFoldResult> indices =
        getAsOpFoldResult(loadStoreLikeOp.getIndices());
    if (llvm::all_of(indices, isZeroInteger)) {
      return rewriter.notifyMatchFailure(
          loadStoreLikeOp, "no computation to extract: offsets are 0s");
    }

    // Create the array of ones of the right size.
    SmallVector<OpFoldResult> ones(loadStoreRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes =
        getViewSizeForEachDim(rewriter, loadStoreLikeOp);
    assert(sizes.size() == loadStoreRank &&
           "Expected one size per load dimension");
    Location loc = loadStoreLikeOp.getLoc();
    // The subview inherits its strides from the original memref and will
    // apply them properly to the input indices.
    // Therefore the strides multipliers are simply ones.
    auto subview =
        rewriter.create<memref::SubViewOp>(loc, /*source=*/srcMemRef,
                                           /*offsets=*/indices,
                                           /*sizes=*/sizes, /*strides=*/ones);
    // Rewrite the load/store with the subview as the base pointer.
    SmallVector<Value> zeros(loadStoreRank,
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));
    LoadStoreLikeOp newLoadStore = rebuildOpFromAddressAndIndices(
        rewriter, loadStoreLikeOp, subview.getResult(), zeros);
    rewriter.replaceOp(loadStoreLikeOp, newLoadStore->getResults());
    return success();
  }
};
} // namespace

void memref::populateExtractAddressComputationsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      LoadStoreLikeOpRewriter<
          memref::LoadOp,
          /*getSrcMemRef=*/getLoadOpSrcMemRef,
          /*rebuildOpFromAddressAndIndices=*/rebuildLoadOp,
          /*getViewSizeForEachDim=*/getLoadOpViewSizeForEachDim>,
      LoadStoreLikeOpRewriter<
          memref::StoreOp,
          /*getSrcMemRef=*/getStoreOpSrcMemRef,
          /*rebuildOpFromAddressAndIndices=*/rebuildStoreOp,
          /*getViewSizeForEachDim=*/getStoreOpViewSizeForEachDim>,
      LoadStoreLikeOpRewriter<
          nvgpu::LdMatrixOp,
          /*getSrcMemRef=*/getLdMatrixOpSrcMemRef,
          /*rebuildOpFromAddressAndIndices=*/rebuildLdMatrixOp>,
      LoadStoreLikeOpRewriter<
          vector::TransferReadOp,
          /*getSrcMemRef=*/getTransferLikeOpSrcMemRef<vector::TransferReadOp>,
          /*rebuildOpFromAddressAndIndices=*/rebuildTransferReadOp>,
      LoadStoreLikeOpRewriter<
          vector::TransferWriteOp,
          /*getSrcMemRef=*/getTransferLikeOpSrcMemRef<vector::TransferWriteOp>,
          /*rebuildOpFromAddressAndIndices=*/rebuildTransferWriteOp>>(
      patterns.getContext());
}
