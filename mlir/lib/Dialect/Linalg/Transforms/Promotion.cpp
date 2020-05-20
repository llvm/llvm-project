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

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::scf;

using llvm::SetVector;

using folded_affine_min = FoldedValueBuilder<AffineMinOp>;
using folded_linalg_range = FoldedValueBuilder<linalg::RangeOp>;
using folded_std_dim = FoldedValueBuilder<DimOp>;
using folded_std_subview = FoldedValueBuilder<SubViewOp>;
using folded_std_view = FoldedValueBuilder<ViewOp>;

#define DEBUG_TYPE "linalg-promotion"

namespace {

/// Helper struct that captures the information required to apply the
/// transformation on each op. This bridges the abstraction gap with the
/// user-facing API which exposes positional arguments to control which operands
/// are promoted.
struct LinalgOpInstancePromotionOptions {
  LinalgOpInstancePromotionOptions(LinalgOp op,
                                   const LinalgPromotionOptions &options);
  /// SubViews to promote.
  SetVector<Value> subViews;
  /// True if the full view should be used for the promoted buffer.
  DenseMap<Value, bool> useFullTileBuffers;
  /// Allow the use of dynamicaly-sized buffers.
  bool dynamicBuffers;
  /// Alignment of promoted buffer.
  Optional<unsigned> alignment;
};
} // namespace

LinalgOpInstancePromotionOptions::LinalgOpInstancePromotionOptions(
    LinalgOp linalgOp, const LinalgPromotionOptions &options)
    : subViews(), useFullTileBuffers(), dynamicBuffers(options.dynamicBuffers),
      alignment(options.alignment) {
  unsigned nBuffers = linalgOp.getNumInputsAndOutputBuffers();
  auto vUseFullTileBuffers =
      options.useFullTileBuffers.getValueOr(llvm::SmallBitVector());
  vUseFullTileBuffers.resize(nBuffers, options.useFullTileBuffersDefault);

  if (options.operandsToPromote.hasValue()) {
    for (auto it : llvm::enumerate(options.operandsToPromote.getValue())) {
      auto *op = linalgOp.getBuffer(it.value()).getDefiningOp();
      if (auto sv = dyn_cast_or_null<SubViewOp>(op)) {
        subViews.insert(sv);
        useFullTileBuffers[sv] = vUseFullTileBuffers[it.index()];
      }
    }
  } else {
    for (unsigned idx = 0; idx < nBuffers; ++idx) {
      auto *op = linalgOp.getBuffer(idx).getDefiningOp();
      if (auto sv = dyn_cast_or_null<SubViewOp>(op)) {
        subViews.insert(sv);
        useFullTileBuffers[sv] = vUseFullTileBuffers[idx];
      }
    }
  }
}

/// If `size` comes from an AffineMinOp and one of the values of AffineMinOp
/// is a constant then return a new value set to the smallest such constant.
/// Otherwise return size.
static Value extractSmallestConstantBoundingSize(OpBuilder &b, Location loc,
                                                 Value size) {
  auto affineMinOp = size.getDefiningOp<AffineMinOp>();
  if (!affineMinOp)
    return size;
  int64_t minConst = std::numeric_limits<int64_t>::max();
  for (auto e : affineMinOp.getAffineMap().getResults())
    if (auto cst = e.dyn_cast<AffineConstantExpr>())
      minConst = std::min(minConst, cst.getValue());
  return (minConst == std::numeric_limits<int64_t>::max())
             ? size
             : b.create<ConstantIndexOp>(loc, minConst);
}

/// Alloc a new buffer of `size`. If `dynamicBuffers` is true allocate exactly
/// the size needed, otherwise try to allocate a static bounding box.
static Value allocBuffer(Type elementType, Value size, bool dynamicBuffers,
                         OperationFolder *folder,
                         Optional<unsigned> alignment = None) {
  auto *ctx = size.getContext();
  auto width = llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
  IntegerAttr alignment_attr;
  if (alignment.hasValue())
    alignment_attr =
        IntegerAttr::get(IntegerType::get(64, ctx), alignment.getValue());
  if (!dynamicBuffers)
    if (auto cst = size.getDefiningOp<ConstantIndexOp>())
      return std_alloc(
          MemRefType::get(width * cst.getValue(), IntegerType::get(8, ctx)),
          ValueRange{}, alignment_attr);
  Value mul =
      folded_std_muli(folder, folded_std_constant_index(folder, width), size);
  return std_alloc(MemRefType::get(-1, IntegerType::get(8, ctx)), mul,
                   alignment_attr);
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
static PromotionInfo promoteSubviewAsNewBuffer(OpBuilder &b, Location loc,
                                               SubViewOp subView,
                                               bool dynamicBuffers,
                                               Optional<unsigned> alignment,
                                               OperationFolder *folder) {
  auto zero = folded_std_constant_index(folder, 0);
  auto one = folded_std_constant_index(folder, 1);

  auto viewType = subView.getType();
  auto rank = viewType.getRank();
  Value allocSize = one;
  SmallVector<Value, 8> fullSizes, partialSizes;
  fullSizes.reserve(rank);
  partialSizes.reserve(rank);
  for (auto en : llvm::enumerate(subView.getOrCreateRanges(b, loc))) {
    auto rank = en.index();
    auto rangeValue = en.value();
    // Try to extract a tight constant.
    LLVM_DEBUG(llvm::dbgs() << "Extract tightest: " << rangeValue.size << "\n");
    Value size = extractSmallestConstantBoundingSize(b, loc, rangeValue.size);
    LLVM_DEBUG(llvm::dbgs() << "Extracted tightest: " << size << "\n");
    allocSize = folded_std_muli(folder, allocSize, size);
    fullSizes.push_back(size);
    partialSizes.push_back(folded_std_dim(folder, subView, rank));
  }
  SmallVector<int64_t, 4> dynSizes(fullSizes.size(), -1);
  auto buffer = allocBuffer(viewType.getElementType(), allocSize,
                            dynamicBuffers, folder, alignment);
  auto fullLocalView = folded_std_view(
      folder, MemRefType::get(dynSizes, viewType.getElementType()), buffer,
      zero, fullSizes);
  SmallVector<Value, 4> zeros(fullSizes.size(), zero);
  SmallVector<Value, 4> ones(fullSizes.size(), one);
  auto partialLocalView =
      folded_std_subview(folder, fullLocalView, zeros, partialSizes, ones);
  return PromotionInfo{buffer, fullLocalView, partialLocalView};
}

static SmallVector<PromotionInfo, 8>
promoteSubViews(OpBuilder &b, Location loc,
                LinalgOpInstancePromotionOptions options,
                OperationFolder *folder) {
  if (options.subViews.empty())
    return {};

  ScopedContext scope(b, loc);
  SmallVector<PromotionInfo, 8> res;
  res.reserve(options.subViews.size());
  DenseMap<Value, PromotionInfo> promotionInfoMap;
  for (auto v : options.subViews) {
    SubViewOp subView = cast<SubViewOp>(v.getDefiningOp());
    auto promotionInfo = promoteSubviewAsNewBuffer(
        b, loc, subView, options.dynamicBuffers, options.alignment, folder);
    promotionInfoMap.insert(std::make_pair(subView.getResult(), promotionInfo));
    res.push_back(promotionInfo);
  }

  for (auto v : options.subViews) {
    SubViewOp subView = cast<SubViewOp>(v.getDefiningOp());
    auto info = promotionInfoMap.find(v);
    if (info == promotionInfoMap.end())
      continue;
    // Only fill the buffer if the full local view is used
    if (!options.useFullTileBuffers[v])
      continue;
    Value fillVal;
    if (auto t = subView.getType().getElementType().dyn_cast<FloatType>())
      fillVal = folded_std_constant(folder, FloatAttr::get(t, 0.0));
    else if (auto t =
                 subView.getType().getElementType().dyn_cast<IntegerType>())
      fillVal = folded_std_constant_int(folder, 0, t);
    // TODO(ntv): fill is only necessary if `promotionInfo` has a full local
    // view that is different from the partial local view and we are on the
    // boundary.
    linalg_fill(info->second.fullLocalView, fillVal);
  }

  for (auto v : options.subViews) {
    auto info = promotionInfoMap.find(v);
    if (info == promotionInfoMap.end())
      continue;
    linalg_copy(cast<SubViewOp>(v.getDefiningOp()),
                info->second.partialLocalView);
  }
  return res;
}

static void promoteSubViews(OpBuilder &b, LinalgOp op,
                            LinalgOpInstancePromotionOptions options,
                            OperationFolder *folder) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");

  if (auto convOp = dyn_cast<linalg::ConvOp>(op.getOperation())) {
    // TODO(ntv): add a level of indirection to linalg.generic.
    if (convOp.padding())
      llvm_unreachable("Unexpected conv with padding");
  }

  // 1. Promote the specified views and use them in the new op.
  auto loc = op.getLoc();
  auto promotedBufferAndViews = promoteSubViews(b, loc, options, folder);
  SmallVector<Value, 8> opViews;
  opViews.reserve(op.getNumInputsAndOutputs());
  SmallVector<std::pair<Value, Value>, 8> writebackViews;
  writebackViews.reserve(promotedBufferAndViews.size());
  unsigned promotedIdx = 0;
  for (auto view : op.getInputsAndOutputBuffers()) {
    if (options.subViews.count(view) != 0) {
      if (options.useFullTileBuffers[view])
        opViews.push_back(promotedBufferAndViews[promotedIdx].fullLocalView);
      else
        opViews.push_back(promotedBufferAndViews[promotedIdx].partialLocalView);
      writebackViews.emplace_back(std::make_pair(
          view, promotedBufferAndViews[promotedIdx].partialLocalView));
      promotedIdx++;
    } else {
      opViews.push_back(view);
    }
  }

  // 2. Append all other operands as they appear, this enforces that such
  // operands are not views. This is to support cases such as FillOp taking
  // extra scalars etc.
  // Keep a reference to output buffers;
  DenseSet<Value> originalOutputs(op.getOutputBuffers().begin(),
                                  op.getOutputBuffers().end());
  op.getOperation()->setOperands(0, opViews.size(), opViews);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(op);
  ScopedContext scope(b, loc);
  // 3. Emit write-back for the promoted output views: copy the partial view.
  for (auto viewAndPartialLocalView : writebackViews)
    if (originalOutputs.count(viewAndPartialLocalView.first))
      linalg_copy(viewAndPartialLocalView.second,
                  viewAndPartialLocalView.first);

  // 4. Dealloc all local buffers.
  for (const auto &pi : promotedBufferAndViews)
    std_dealloc(pi.buffer);
}

LogicalResult
mlir::linalg::promoteSubviewsPrecondition(Operation *op,
                                          LinalgPromotionOptions options) {
  LinalgOp linOp = dyn_cast<LinalgOp>(op);
  // Transformation applies to buffers only.
  if (!linOp || !linOp.hasBufferSemantics())
    return failure();
  // Check that at least one of the requested operands is indeed a subview.
  for (auto en : llvm::enumerate(linOp.getInputsAndOutputBuffers())) {
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

LinalgOp mlir::linalg::promoteSubViews(OpBuilder &b, LinalgOp linalgOp,
                                       LinalgPromotionOptions options,
                                       OperationFolder *folder) {
  LinalgOpInstancePromotionOptions linalgOptions(linalgOp, options);
  ::promoteSubViews(
      b, linalgOp, LinalgOpInstancePromotionOptions(linalgOp, options), folder);
  return linalgOp;
}

namespace {
struct LinalgPromotionPass : public LinalgPromotionBase<LinalgPromotionPass> {
  LinalgPromotionPass() = default;
  LinalgPromotionPass(bool dynamicBuffers) {
    this->dynamicBuffers = dynamicBuffers;
  }

  void runOnFunction() override {
    OperationFolder folder(&getContext());
    getFunction().walk([this, &folder](LinalgOp op) {
      auto options = LinalgPromotionOptions().setDynamicBuffers(dynamicBuffers);
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
mlir::createLinalgPromotionPass(bool dynamicBuffers) {
  return std::make_unique<LinalgPromotionPass>(dynamicBuffers);
}
std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgPromotionPass() {
  return std::make_unique<LinalgPromotionPass>();
}
