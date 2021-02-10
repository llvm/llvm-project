//===- Promotion.cpp - Implementation of linalg Promotion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the linalg dialect Promotion pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/FoldedIntrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::scf;

using llvm::MapVector;

using folded_affine_min = FoldedValueBuilder<AffineMinOp>;
using folded_linalg_range = FoldedValueBuilder<linalg::RangeOp>;
using folded_std_dim = FoldedValueBuilder<DimOp>;
using folded_std_subview = FoldedValueBuilder<SubViewOp>;
using folded_memref_view = FoldedValueBuilder<memref::ViewOp>;

#define DEBUG_TYPE "linalg-promotion"

/// Alloc a new buffer of `size`. If `dynamicBuffers` is true allocate exactly
/// the size needed, otherwise try to allocate a static bounding box.
static Value allocBuffer(const LinalgPromotionOptions &options,
                         Type elementType, Value size, bool dynamicBuffers,
                         OperationFolder *folder,
                         Optional<unsigned> alignment = None) {
  auto *ctx = size.getContext();
  auto width = llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
  IntegerAttr alignment_attr;
  if (alignment.hasValue())
    alignment_attr =
        IntegerAttr::get(IntegerType::get(ctx, 64), alignment.getValue());
  if (!dynamicBuffers)
    if (auto cst = size.getDefiningOp<ConstantIndexOp>())
      return options.useAlloca
                 ? memref_alloca(MemRefType::get(width * cst.getValue(),
                                                 IntegerType::get(ctx, 8)),
                                 ValueRange{}, alignment_attr)
                       .value
                 : memref_alloc(MemRefType::get(width * cst.getValue(),
                                                IntegerType::get(ctx, 8)),
                                ValueRange{}, alignment_attr)
                       .value;
  Value mul =
      folded_std_muli(folder, folded_std_constant_index(folder, width), size);
  return options.useAlloca
             ? memref_alloca(MemRefType::get(-1, IntegerType::get(ctx, 8)), mul,
                             alignment_attr)
                   .value
             : memref_alloc(MemRefType::get(-1, IntegerType::get(ctx, 8)), mul,
                            alignment_attr)
                   .value;
}

/// Default allocation callback function. This allocates a promoted buffer when
/// no call back to do so is provided. The default is to allocate a
/// memref<..xi8> and return a view to get a memref type of shape
/// boundingSubViewSize.
static Optional<Value> defaultAllocBufferCallBack(
    const LinalgPromotionOptions &options, OpBuilder &builder,
    SubViewOp subView, ArrayRef<Value> boundingSubViewSize, bool dynamicBuffers,
    Optional<unsigned> alignment, OperationFolder *folder) {
  ShapedType viewType = subView.getType();
  int64_t rank = viewType.getRank();
  (void)rank;
  assert(rank > 0 && boundingSubViewSize.size() == static_cast<size_t>(rank));
  auto zero = folded_std_constant_index(folder, 0);
  auto one = folded_std_constant_index(folder, 1);

  Value allocSize = one;
  for (auto size : llvm::enumerate(boundingSubViewSize))
    allocSize = folded_std_muli(folder, allocSize, size.value());
  Value buffer = allocBuffer(options, viewType.getElementType(), allocSize,
                             dynamicBuffers, folder, alignment);
  SmallVector<int64_t, 4> dynSizes(boundingSubViewSize.size(),
                                   ShapedType::kDynamicSize);
  Value view = folded_memref_view(
      folder, MemRefType::get(dynSizes, viewType.getElementType()), buffer,
      zero, boundingSubViewSize);
  return view;
}

/// Default implementation of deallocation of the buffer use for promotion. It
/// expects to get the same value that the default allocation method returned,
/// i.e. result of a ViewOp.
static LogicalResult
defaultDeallocBufferCallBack(const LinalgPromotionOptions &options,
                             OpBuilder &b, Value fullLocalView) {
  auto viewOp = fullLocalView.getDefiningOp<memref::ViewOp>();
  assert(viewOp && "expected full local view to be a ViewOp");
  if (!options.useAlloca)
    memref_dealloc(viewOp.source());
  return success();
}

namespace {

/// Helper struct that captures the information required to apply the
/// transformation on each op. This bridges the abstraction gap with the
/// user-facing API which exposes positional arguments to control which operands
/// are promoted.
struct LinalgOpInstancePromotionOptions {
  LinalgOpInstancePromotionOptions(LinalgOp op,
                                   const LinalgPromotionOptions &options);
  /// SubViews to promote.
  MapVector<unsigned, Value> subViews;
  /// True if the full view should be used for the promoted buffer.
  DenseMap<Value, bool> useFullTileBuffers;

  /// Callback functions for allocation and deallocation of promoted buffers, as
  /// well as to copy the data into and out of these buffers.
  AllocBufferCallbackFn allocationFn;
  DeallocBufferCallbackFn deallocationFn;
  CopyCallbackFn copyInFn;
  CopyCallbackFn copyOutFn;

  /// Allow the use of dynamically-sized buffers.
  bool dynamicBuffers;
  /// Alignment of promoted buffer.
  Optional<unsigned> alignment;
};
} // namespace

LinalgOpInstancePromotionOptions::LinalgOpInstancePromotionOptions(
    LinalgOp linalgOp, const LinalgPromotionOptions &options)
    : subViews(), dynamicBuffers(options.dynamicBuffers),
      alignment(options.alignment) {
  assert(linalgOp.hasBufferSemantics() && "revisit usage of shaped operand");
  unsigned nBuffers = linalgOp.getNumShapedOperands();
  auto vUseFullTileBuffers =
      options.useFullTileBuffers.getValueOr(llvm::SmallBitVector());
  vUseFullTileBuffers.resize(nBuffers, options.useFullTileBuffersDefault);

  for (unsigned idx = 0; idx != nBuffers; ++idx) {
    if (options.operandsToPromote && !options.operandsToPromote->count(idx))
      continue;
    auto *op = linalgOp.getShapedOperand(idx).getDefiningOp();
    if (auto sv = dyn_cast_or_null<SubViewOp>(op)) {
      subViews[idx] = sv;
      useFullTileBuffers[sv] = vUseFullTileBuffers[idx];
    }
  }

  allocationFn =
      (options.allocationFn ? *(options.allocationFn)
                            : [&](OpBuilder &builder, SubViewOp subViewOp,
                                  ArrayRef<Value> boundingSubViewSize,
                                  OperationFolder *folder) -> Optional<Value> {
        return defaultAllocBufferCallBack(options, builder, subViewOp,
                                          boundingSubViewSize, dynamicBuffers,
                                          alignment, folder);
      });
  deallocationFn =
      (options.deallocationFn
           ? *(options.deallocationFn)
           : [&](OpBuilder &b, Value buffer) {
               return defaultDeallocBufferCallBack(options, b, buffer);
             });
  auto defaultCopyCallBack = [&](OpBuilder &builder, Value src,
                                 Value dst) -> LogicalResult {
    linalg_copy(src, dst);
    return success();
  };
  copyInFn = (options.copyInFn ? *(options.copyInFn) : defaultCopyCallBack);
  copyOutFn = (options.copyOutFn ? *(options.copyOutFn) : defaultCopyCallBack);
}

// Performs promotion of a `subView` into a local buffer of the size of the
// *ranges* of the `subView`. This produces a buffer whose size may be bigger
// than the actual size of the `subView` at the boundaries.
// This is related to the full/partial tile problem.
// Returns a PromotionInfo containing a `buffer`, `fullLocalView` and
// `partialLocalView` such that:
//   * `buffer` is always the size of the full tile.
//   * `fullLocalView` is a dense contiguous view into that buffer.
//   * `partialLocalView` is a dense non-contiguous slice of `fullLocalView`
//     that corresponds to the size of `subView` and accounting for boundary
//     effects.
// The point of the full tile buffer is that constant static tile sizes are
// folded and result in a buffer type with statically known size and alignment
// properties.
// To account for general boundary effects, padding must be performed on the
// boundary tiles. For now this is done with an unconditional `fill` op followed
// by a partial `copy` op.
Optional<PromotionInfo> mlir::linalg::promoteSubviewAsNewBuffer(
    OpBuilder &b, Location loc, SubViewOp subView,
    AllocBufferCallbackFn allocationFn, OperationFolder *folder) {
  ScopedContext scopedContext(b, loc);
  auto viewType = subView.getType();
  auto rank = viewType.getRank();
  SmallVector<Value, 4> fullSizes;
  SmallVector<OpFoldResult> partialSizes;
  fullSizes.reserve(rank);
  partialSizes.reserve(rank);
  for (auto en : llvm::enumerate(subView.getOrCreateRanges(b, loc))) {
    auto rangeValue = en.value();
    // Try to extract a tight constant.
    LLVM_DEBUG(llvm::dbgs() << "Extract tightest: " << rangeValue.size << "\n");
    IntegerAttr sizeAttr = getSmallestBoundingIndex(rangeValue.size);
    Value size =
        (!sizeAttr) ? rangeValue.size : b.create<ConstantOp>(loc, sizeAttr);
    LLVM_DEBUG(llvm::dbgs() << "Extracted tightest: " << size << "\n");
    fullSizes.push_back(size);
    partialSizes.push_back(folded_std_dim(folder, subView, en.index()).value);
  }
  SmallVector<int64_t, 4> dynSizes(fullSizes.size(), -1);
  // If a callback is not specified, then use the default implementation for
  // allocating the promoted buffer.
  Optional<Value> fullLocalView = allocationFn(b, subView, fullSizes, folder);
  if (!fullLocalView)
    return {};
  SmallVector<OpFoldResult, 4> zeros(fullSizes.size(), b.getIndexAttr(0));
  SmallVector<OpFoldResult, 4> ones(fullSizes.size(), b.getIndexAttr(1));
  auto partialLocalView =
      folded_std_subview(folder, *fullLocalView, zeros, partialSizes, ones);
  return PromotionInfo{*fullLocalView, partialLocalView};
}

static Optional<MapVector<unsigned, PromotionInfo>>
promoteSubViews(OpBuilder &b, Location loc,
                LinalgOpInstancePromotionOptions options,
                OperationFolder *folder) {
  if (options.subViews.empty())
    return {};

  ScopedContext scope(b, loc);
  MapVector<unsigned, PromotionInfo> promotionInfoMap;

  for (auto v : options.subViews) {
    SubViewOp subView = cast<SubViewOp>(v.second.getDefiningOp());
    Optional<PromotionInfo> promotionInfo = promoteSubviewAsNewBuffer(
        b, loc, subView, options.allocationFn, folder);
    if (!promotionInfo)
      return {};
    promotionInfoMap[v.first] = *promotionInfo;

    // Only fill the buffer if the full local view is used
    if (!options.useFullTileBuffers[v.second])
      continue;
    Value fillVal;
    if (auto t = subView.getType().getElementType().dyn_cast<FloatType>())
      fillVal = folded_std_constant(folder, FloatAttr::get(t, 0.0));
    else if (auto t =
                 subView.getType().getElementType().dyn_cast<IntegerType>())
      fillVal = folded_std_constant_int(folder, 0, t);
    linalg_fill(promotionInfo->fullLocalView, fillVal);
  }

  // Copy data into the promoted buffers. Use callback if provided.
  for (auto v : options.subViews) {
    auto info = promotionInfoMap.find(v.first);
    if (info == promotionInfoMap.end())
      continue;
    if (failed(options.copyInFn(b, cast<SubViewOp>(v.second.getDefiningOp()),
                                info->second.partialLocalView)))
      return {};
  }
  return promotionInfoMap;
}

static Optional<LinalgOp>
promoteSubViews(OpBuilder &b, LinalgOp op,
                LinalgOpInstancePromotionOptions options,
                OperationFolder *folder) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");

  if (auto convOp = dyn_cast<linalg::ConvOp>(op.getOperation())) {
    // TODO: add a level of indirection to linalg.generic.
    if (convOp.padding())
      return {};
  }

  // 1. Promote the specified views and use them in the new op.
  auto loc = op.getLoc();
  auto promotedBuffersAndViews = promoteSubViews(b, loc, options, folder);
  if (!promotedBuffersAndViews ||
      promotedBuffersAndViews->size() != options.subViews.size())
    return {};

  // 2. Append all other operands as they appear, this enforces that such
  // operands are not views. This is to support cases such as FillOp taking
  // extra scalars etc.  Keep a reference to output buffers;
  SmallVector<Value, 8> opViews;
  opViews.reserve(op.getNumShapedOperands());
  SmallVector<std::pair<Value, Value>, 8> writebackViews;
  writebackViews.reserve(promotedBuffersAndViews->size());
  for (auto view : llvm::enumerate(op.getShapedOperands())) {
    if (options.subViews.count(view.index()) != 0) {
      if (options.useFullTileBuffers[view.value()])
        opViews.push_back(
            (*promotedBuffersAndViews)[view.index()].fullLocalView);
      else
        opViews.push_back(
            (*promotedBuffersAndViews)[view.index()].partialLocalView);
      if (view.index() >= op.getNumInputs())
        writebackViews.emplace_back(std::make_pair(
            view.value(),
            (*promotedBuffersAndViews)[view.index()].partialLocalView));
    } else {
      opViews.push_back(view.value());
    }
  }
  op->setOperands(0, opViews.size(), opViews);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(op);
  ScopedContext scope(b, loc);
  // 3. Emit write-back for the promoted output views: copy the partial view.
  for (auto viewAndPartialLocalView : writebackViews) {
    if (failed(options.copyOutFn(b, viewAndPartialLocalView.second,
                                 viewAndPartialLocalView.first)))
      return {};
  }

  // 4. Dealloc all local buffers.
  for (const auto &pi : *promotedBuffersAndViews)
    (void)options.deallocationFn(b, pi.second.fullLocalView);
  return op;
}

LogicalResult
mlir::linalg::promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options) {
  LinalgOp linOp = dyn_cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linOp || !linOp.hasBufferSemantics())
    return failure();
  // Check that at least one of the requested operands is indeed a subview.
  for (auto en : llvm::enumerate(linOp.getShapedOperands())) {
    auto sv = isa_and_nonnull<SubViewOp>(en.value().getDefiningOp());
    if (sv) {
      if (!options.operandsToPromote.hasValue() ||
          options.operandsToPromote->count(en.index()))
        return success();
    }
  }
  // TODO: Check all subviews requested are bound by a static constant.
  // TODO: Check that the total footprint fits within a given size.
  return failure();
}

Optional<LinalgOp> mlir::linalg::promoteSubViews(OpBuilder &b,
                                                 LinalgOp linalgOp,
                                                 LinalgPromotionOptions options,
                                                 OperationFolder *folder) {
  LinalgOpInstancePromotionOptions linalgOptions(linalgOp, options);
  return ::promoteSubViews(
      b, linalgOp, LinalgOpInstancePromotionOptions(linalgOp, options), folder);
}

namespace {
struct LinalgPromotionPass : public LinalgPromotionBase<LinalgPromotionPass> {
  LinalgPromotionPass() = default;
  LinalgPromotionPass(bool dynamicBuffers, bool useAlloca) {
    this->dynamicBuffers = dynamicBuffers;
    this->useAlloca = useAlloca;
  }

  void runOnFunction() override {
    OperationFolder folder(&getContext());
    getFunction().walk([this, &folder](LinalgOp op) {
      auto options = LinalgPromotionOptions()
                         .setDynamicBuffers(dynamicBuffers)
                         .setUseAlloca(useAlloca);
      if (failed(promoteSubviewsPrecondition(op, options)))
        return;
      LLVM_DEBUG(llvm::dbgs() << "Promote: " << *(op.getOperation()) << "\n");
      OpBuilder b(op);
      promoteSubViews(b, op, options, &folder);
    });
  }
};
} // namespace

// TODO: support more transformation options in the pass.
std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgPromotionPass(bool dynamicBuffers, bool useAlloca) {
  return std::make_unique<LinalgPromotionPass>(dynamicBuffers, useAlloca);
}
std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgPromotionPass() {
  return std::make_unique<LinalgPromotionPass>();
}
