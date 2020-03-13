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

#include "mlir/Dialect/AffineOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/FoldUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::linalg;
using namespace mlir::loop;

using llvm::SetVector;

using folded_affine_min = folded::ValueBuilder<AffineMinOp>;
using folded_linalg_range = folded::ValueBuilder<linalg::RangeOp>;

#define DEBUG_TYPE "linalg-promotion"

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");
static llvm::cl::opt<bool> clPromoteDynamic(
    "test-linalg-promote-dynamic",
    llvm::cl::desc("Test generation of dynamic promoted buffers"),
    llvm::cl::cat(clOptionsCategory), llvm::cl::init(false));

static Value allocBuffer(Type elementType, Value size, bool dynamicBuffers) {
  auto *ctx = size.getContext();
  auto width = llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
  if (!dynamicBuffers)
    if (auto cst = dyn_cast_or_null<ConstantIndexOp>(size.getDefiningOp()))
      return std_alloc(
          MemRefType::get(width * cst.getValue(), IntegerType::get(8, ctx)));
  Value mul = std_muli(std_constant_index(width), size);
  return std_alloc(MemRefType::get(-1, IntegerType::get(8, ctx)), mul);
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
static PromotionInfo promoteFullTileBuffer(OpBuilder &b, Location loc,
                                           SubViewOp subView,
                                           bool dynamicBuffers,
                                           OperationFolder *folder) {
  auto zero = folded_std_constant_index(folder, 0);
  auto one = folded_std_constant_index(folder, 1);

  auto viewType = subView.getType();
  auto rank = viewType.getRank();
  Value allocSize = one;
  SmallVector<Value, 8> fullRanges, partialRanges;
  fullRanges.reserve(rank);
  partialRanges.reserve(rank);
  for (auto en : llvm::enumerate(subView.getRanges())) {
    auto rank = en.index();
    auto rangeValue = en.value();
    Value d = rangeValue.size;
    allocSize = folded_std_muli(folder, allocSize, d).getValue();
    fullRanges.push_back(d);
    partialRanges.push_back(
        folded_linalg_range(folder, zero, std_dim(subView, rank), one));
  }
  SmallVector<int64_t, 4> dynSizes(fullRanges.size(), -1);
  auto buffer =
      allocBuffer(viewType.getElementType(), allocSize, dynamicBuffers);
  auto fullLocalView = std_view(
      MemRefType::get(dynSizes, viewType.getElementType()), buffer, fullRanges);
  auto partialLocalView = linalg_slice(fullLocalView, partialRanges);
  return PromotionInfo{buffer, fullLocalView, partialLocalView};
}

SmallVector<PromotionInfo, 8>
mlir::linalg::promoteSubViews(OpBuilder &b, Location loc,
                              ArrayRef<Value> subViews, bool dynamicBuffers,
                              OperationFolder *folder) {
  if (subViews.empty())
    return {};

  ScopedContext scope(b, loc);
  SmallVector<PromotionInfo, 8> res;
  res.reserve(subViews.size());
  DenseMap<Value, PromotionInfo> promotionInfoMap;
  for (auto v : subViews) {
    SubViewOp subView = cast<SubViewOp>(v.getDefiningOp());
    auto promotionInfo =
        promoteFullTileBuffer(b, loc, subView, dynamicBuffers, folder);
    promotionInfoMap.insert(std::make_pair(subView.getResult(), promotionInfo));
    res.push_back(promotionInfo);
  }

  for (auto v : subViews) {
    SubViewOp subView = cast<SubViewOp>(v.getDefiningOp());
    auto info = promotionInfoMap.find(v);
    if (info == promotionInfoMap.end())
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

  for (auto v : subViews) {
    auto info = promotionInfoMap.find(v);
    if (info == promotionInfoMap.end())
      continue;
    linalg_copy(cast<SubViewOp>(v.getDefiningOp()),
                info->second.partialLocalView);
  }
  return res;
}

LinalgOp mlir::linalg::promoteSubViewOperands(OpBuilder &b, LinalgOp op,
                                              SetVector<Value> subViews,
                                              bool dynamicBuffers,
                                              OperationFolder *folder) {
  assert(op.hasBufferSemantics() && "expected linalg op with buffer semantics");

  if (auto convOp = dyn_cast<linalg::ConvOp>(op.getOperation())) {
    // TODO(ntv): add a level of indirection to linalg.generic.
    if (convOp.padding())
      llvm_unreachable("Unexpected conv with padding");
  }

  // 1. Promote the specified views and use them in the new op.
  ScopedContext scope(b, op.getLoc());
  auto promotedBufferAndViews = promoteSubViews(
      b, op.getLoc(), subViews.getArrayRef(), dynamicBuffers, folder);
  SmallVector<Value, 8> opViews;
  opViews.reserve(op.getNumInputsAndOutputs());
  SmallVector<std::pair<Value, Value>, 8> writebackViews;
  writebackViews.reserve(subViews.size());
  unsigned promotedIdx = 0;
  for (auto view : op.getInputsAndOutputBuffers()) {
    if (subViews.count(view) != 0) {
      opViews.push_back(promotedBufferAndViews[promotedIdx].fullLocalView);
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
  auto operands = getAssumedNonViewOperands(op);
  opViews.append(operands.begin(), operands.end());
  LinalgOp res = op.clone(b, op.getLoc(), opViews);

  // 3. Emit write-back for the promoted output views: copy the partial view.
  for (auto viewAndPartialLocalView : writebackViews) {
    // WARNING: MUST use the old op to determine whether the operand view is an
    // output.
    bool isOutput =
        op.getIndexOfOutputBuffer(viewAndPartialLocalView.first).hasValue();
    if (isOutput)
      linalg_copy(viewAndPartialLocalView.second,
                  viewAndPartialLocalView.first);
  }

  // 4. Dealloc local buffers.
  for (const auto &pi : promotedBufferAndViews)
    std_dealloc(pi.buffer);

  return res;
}

static void promoteSubViews(FuncOp f, bool dynamicBuffers) {
  SmallVector<LinalgOp, 8> toErase;
  OperationFolder folder(f.getContext());
  f.walk([dynamicBuffers, &folder, &toErase](LinalgOp op) {
    if (!op.hasBufferSemantics())
      return;

    // TODO(ntv) some heuristic here to decide what to promote. Atm only float
    // and integer buffers can be promoted.
    SetVector<Value> subViews;
    OpBuilder b(op);
    for (auto it : op.getInputsAndOutputBuffers())
      if (auto sv = dyn_cast_or_null<SubViewOp>(it.getDefiningOp()))
        if (sv.getType().getElementType().isSignlessIntOrFloat())
          subViews.insert(sv);
    if (!subViews.empty()) {
      promoteSubViewOperands(b, op, subViews, dynamicBuffers, &folder);
      toErase.push_back(op);
    }
  });
  for (auto op : toErase)
    op.erase();
}

namespace {
struct LinalgPromotionPass : public FunctionPass<LinalgPromotionPass> {
  LinalgPromotionPass() = default;
  LinalgPromotionPass(bool dynamicBuffers) : dynamicBuffers(dynamicBuffers) {}

  void runOnFunction() override {
    promoteSubViews(getFunction(), dynamicBuffers);
  }

  bool dynamicBuffers;
};
} // namespace

std::unique_ptr<OpPassBase<FuncOp>>
mlir::createLinalgPromotionPass(bool dynamicBuffers) {
  return std::make_unique<LinalgPromotionPass>(dynamicBuffers);
}

static PassRegistration<LinalgPromotionPass>
    pass("linalg-promote-subviews", "promote subview ops to local buffers", [] {
      return std::make_unique<LinalgPromotionPass>(clPromoteDynamic);
    });
