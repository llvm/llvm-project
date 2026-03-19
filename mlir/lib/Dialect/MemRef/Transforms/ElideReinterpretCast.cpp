//===-ElideReinterpretCast.cpp - Expansion patterns for MemRef operations-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>

namespace mlir {
namespace memref {
#define GEN_PASS_DEF_ELIDEREINTERPRETCASTPASS
#include "mlir/Dialect/MemRef/Transforms/Passes.h.inc"
} // namespace memref
} // namespace mlir

using namespace mlir;

namespace {

/// Returns true if `rc` represents a scalar view (all sizes == 1)
/// into a memref that has exactly one non-unit dimension located at
/// either the first or last position (i.e. a "row" or "column").
///
/// Examples that return true:
///
///   // Row-major slice (last dim is non-unit)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1, 1], strides: [1, 1, 1]
///     : memref<1x1x8xi32> to memref<1x1x1xi32>
///
///   // Column-major slice (first dim is non-unit)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [1, 1]
///     : memref<2x1xf32> to memref<1x1xf32>
///
///   // Random strides
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [10, 100]
///     : memref<2x1xf32, strided<[10, 100]>>
///         to memref<1x1xf32>
///
///   // Rank-1 case
///   memref.reinterpret_cast %buf to offset: [%off],
///     sizes: [1], strides: [1]
///     : memref<8xi32> to memref<1xi32>
///
/// Examples that return false:
///
///   // More non-unit dims
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1, 1], strides: [1, 1, 1]
///     : memref<1x2x8xi32> to memref<1x1x1xi32>
///
///   // View is not scalar (size != 1)
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [2, 1], strides: [1, 1]
///     : memref<1x2xf32> to memref<2x1xf32>
///
///   // Base has non-identity layout
///   %buff = memref.alloc() : memref<1x2xf32, strided<[1, 3]>>
///   memref.reinterpret_cast %buff to offset: [%off],
///     sizes: [1, 1], strides: [1, 1]
///     : memref<1x2xf32, strided<[1, 3]>> to memref<1x1xf32>
static bool isScalarSlice(memref::ReinterpretCastOp rc) {
  auto rcInputTy = dyn_cast<MemRefType>(rc.getSource().getType());
  auto rcOutputTy = dyn_cast<MemRefType>(rc.getType());

  // Reject strided base - logic for computing linear idx is TODO
  if (!rcInputTy.getLayout().isIdentity())
    return false;

  // Reject non-matching ranks
  unsigned srcRank = rcInputTy.getRank();
  if (srcRank != rcOutputTy.getRank())
    return false;

  ArrayRef<int64_t> sizes = rc.getStaticSizes();

  // View must be scalar: memref<1x...x1>
  if (!llvm::all_of(rcOutputTy.getShape(),
                    [](int64_t dim) { return dim == 1; }))
    return false;

  // Sizes must all be statically 1
  if (!llvm::all_of(sizes, [](int64_t size) {
        return !ShapedType::isDynamic(size) && size == 1;
      }))
    return false;

  // Rank-1 special case
  if (srcRank == 1) {
    // Reject non-scalar output
    if (rcOutputTy.getDimSize(0) > 1)
      return false;
  }

  int nonUnitCount =
      std::count_if(rcInputTy.getShape().begin(), rcInputTy.getShape().end(),
                    [](int dim) { return dim != 1; });
  return nonUnitCount == 1;
}

/// Rewrites `memref.copy` of a 1-element MemRef as a scalar load-store pair
///
/// The pattern matches a reinterpret_cast that creates a scalar view
/// (`sizes = [1, ..., 1]`) into a memref with a single non-unit dimension.
/// Since the view contains only one element, the accessed address is
/// determined solely by the base pointer and the offset.
///
/// Two layouts are supported:
///   * row-major slice  (stride pattern [N, ..., 1])
///   * column-major slice (stride pattern [1, ..., N])
///
/// BEFORE (row-major slice)
///   %view = memref.reinterpret_cast %base
///     to offset: [%off], sizes: [1, ..., 1], strides: [N, ..., 1]
///       : memref<1x...xNxf32>
///         to memref<1x...x1xf32, strided<[N, ..., 1], offset: ?>>
///   memref.copy %src, %view
///     : memref<1x...x1xf32>
///       to memref<1x...x1xf32, strided<[N, ..., 1], offset: ?>>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v  = memref.load %src[%c0, ..., %c0] : memref<1x...x1xf32>
///   memref.store %v, %base[%c0, ..., %off] : memref<1x...xNxf32>
///
/// BEFORE (column-major slice)
///   %view = memref.reinterpret_cast %base
///     to offset: [%off], sizes: [1, ..., 1], strides: [1, ..., N]
///       : memref<Nx...x1xf32>
///         to memref<1x...x1xf32, strided<[1, ..., N], offset: ?>>
///   memref.copy %src, %view
///     : memref<1x...x1xf32>
///       to memref<1x...x1xf32, strided<[1, ..., N], offset: ?>>
///
/// AFTER
///   %c0 = arith.constant 0 : index
///   %v  = memref.load %src[%c0, ..., %c0] : memref<1x...x1xf32>
///   memref.store %v, %base[%off, ..., %c0] : memref<Nx...x1xf32>
struct CopyToScalarLoadAndStore : public OpRewritePattern<memref::CopyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Value rcOutput = op.getTarget();
    auto rc = rcOutput.getDefiningOp<memref::ReinterpretCastOp>();
    if (!rc)
      return rewriter.notifyMatchFailure(
          op, "target is not a memref.reinterpret_cast");

    if (!isScalarSlice(rc))
      return rewriter.notifyMatchFailure(
          op, "reinterpret_cast does not match scalar slice");

    Location loc = op.getLoc();

    Value src = op.getSource();
    Value dst = rc.getSource();

    auto dstType = cast<MemRefType>(dst.getType());
    unsigned dstRank = dstType.getRank();

    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);

    auto srcType = cast<MemRefType>(src.getType());
    SmallVector<Value> loadIndices(srcType.getRank(), zero);
    auto offsets = rc.getMixedOffsets();
    assert(offsets.size() == 1 && "Expecting single offset");
    OpFoldResult offset = offsets[0];
    Value storeOffset = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
    unsigned offsetDim = dstType.getDimSize(0) == 1 ? dstRank - 1 : 0;
    SmallVector<Value> storeIndices(dstRank, zero);
    storeIndices[offsetDim] = storeOffset;
    // If the only user of `rc` is the current Op (which is about to be erased),
    // we can safely erase it.
    if (rcOutput.hasOneUse())
      rewriter.eraseOp(rc);

    Value val = memref::LoadOp::create(rewriter, loc, src, loadIndices);
    memref::StoreOp::create(rewriter, loc, val, dst, storeIndices);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ElideReinterpretCastPass
    : public memref::impl::ElideReinterpretCastPassBase<
          ElideReinterpretCastPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    RewritePatternSet patterns(&ctx);
    memref::populateElideReinterpretCastPatterns(patterns);
    ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<memref::CopyOp>([](memref::CopyOp op) {
      auto rc = op.getTarget().getDefiningOp<memref::ReinterpretCastOp>();
      if (!rc)
        return true;
      return !isScalarSlice(rc);
    });
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::memref::populateElideReinterpretCastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<CopyToScalarLoadAndStore>(patterns.getContext());
}
