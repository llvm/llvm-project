//===- IndependenceTransforms.cpp - Make ops independent of values --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;
using namespace mlir::memref;

/// Make the given OpFoldResult independent of all independencies.
static FailureOr<OpFoldResult> makeIndependent(OpBuilder &b, Location loc,
                                               OpFoldResult ofr,
                                               ValueRange independencies) {
  if (ofr.is<Attribute>())
    return ofr;
  AffineMap boundMap;
  ValueDimList mapOperands;
  if (failed(ValueBoundsConstraintSet::computeIndependentBound(
          boundMap, mapOperands, presburger::BoundType::UB, ofr, independencies,
          /*closedUB=*/true)))
    return failure();
  return affine::materializeComputedBound(b, loc, boundMap, mapOperands);
}

FailureOr<Value> memref::buildIndependentOp(OpBuilder &b,
                                            memref::AllocaOp allocaOp,
                                            ValueRange independencies) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(allocaOp);
  Location loc = allocaOp.getLoc();

  SmallVector<OpFoldResult> newSizes;
  for (OpFoldResult ofr : allocaOp.getMixedSizes()) {
    auto ub = makeIndependent(b, loc, ofr, independencies);
    if (failed(ub))
      return failure();
    newSizes.push_back(*ub);
  }

  // Return existing memref::AllocaOp if nothing has changed.
  if (llvm::equal(allocaOp.getMixedSizes(), newSizes))
    return allocaOp.getResult();

  // Create a new memref::AllocaOp.
  Value newAllocaOp =
      b.create<AllocaOp>(loc, newSizes, allocaOp.getType().getElementType());

  // Create a memref::SubViewOp.
  SmallVector<OpFoldResult> offsets(newSizes.size(), b.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(newSizes.size(), b.getIndexAttr(1));
  return b
      .create<SubViewOp>(loc, newAllocaOp, offsets, allocaOp.getMixedSizes(),
                         strides)
      .getResult();
}

/// Push down an UnrealizedConversionCastOp past a SubViewOp.
static UnrealizedConversionCastOp
propagateSubViewOp(RewriterBase &rewriter,
                   UnrealizedConversionCastOp conversionOp, SubViewOp op) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);
  auto newResultType = cast<MemRefType>(SubViewOp::inferRankReducedResultType(
      op.getType().getShape(), op.getSourceType(), op.getMixedOffsets(),
      op.getMixedSizes(), op.getMixedStrides()));
  Value newSubview = rewriter.create<SubViewOp>(
      op.getLoc(), newResultType, conversionOp.getOperand(0),
      op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides());
  auto newConversionOp = rewriter.create<UnrealizedConversionCastOp>(
      op.getLoc(), op.getType(), newSubview);
  rewriter.replaceAllUsesWith(op.getResult(), newConversionOp->getResult(0));
  return newConversionOp;
}

/// Given an original op and a new, modified op with the same number of results,
/// whose memref return types may differ, replace all uses of the original op
/// with the new op and propagate the new memref types through the IR.
///
/// Example:
/// %from = memref.alloca(%sz) : memref<?xf32>
/// %to = memref.subview ... : ... to memref<?xf32, strided<[1], offset: ?>>
/// memref.store %cst, %from[%c0] : memref<?xf32>
///
/// In the above example, all uses of %from are replaced with %to. This can be
/// done directly for ops such as memref.store. For ops that have memref results
/// (e.g., memref.subview), the result type may depend on the operand type, so
/// we cannot just replace all uses. There is special handling for common memref
/// ops. For all other ops, unrealized_conversion_cast is inserted.
static void replaceAndPropagateMemRefType(RewriterBase &rewriter,
                                          Operation *from, Operation *to) {
  assert(from->getNumResults() == to->getNumResults() &&
         "expected same number of results");
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(to);

  // Wrap new results in unrealized_conversion_cast and replace all uses of the
  // original op.
  SmallVector<UnrealizedConversionCastOp> unrealizedConversions;
  for (const auto &it :
       llvm::enumerate(llvm::zip(from->getResults(), to->getResults()))) {
    unrealizedConversions.push_back(rewriter.create<UnrealizedConversionCastOp>(
        to->getLoc(), std::get<0>(it.value()).getType(),
        std::get<1>(it.value())));
    rewriter.replaceAllUsesWith(from->getResult(it.index()),
                                unrealizedConversions.back()->getResult(0));
  }

  // Push unrealized_conversion_cast ops further down in the IR. I.e., try to
  // wrap results instead of operands in a cast.
  for (int i = 0; i < static_cast<int>(unrealizedConversions.size()); ++i) {
    UnrealizedConversionCastOp conversion = unrealizedConversions[i];
    assert(conversion->getNumOperands() == 1 &&
           conversion->getNumResults() == 1 &&
           "expected single operand and single result");
    SmallVector<Operation *> users = llvm::to_vector(conversion->getUsers());
    for (Operation *user : users) {
      // Handle common memref dialect ops that produce new memrefs and must
      // be recreated with the new result type.
      if (auto subviewOp = dyn_cast<SubViewOp>(user)) {
        unrealizedConversions.push_back(
            propagateSubViewOp(rewriter, conversion, subviewOp));
        continue;
      }

      // TODO: Other memref ops such as memref.collapse_shape/expand_shape
      // should also be handled here.

      // Skip any ops that produce MemRef result or have MemRef region block
      // arguments. These may need special handling (e.g., scf.for).
      if (llvm::any_of(user->getResultTypes(),
                       [](Type t) { return isa<MemRefType>(t); }))
        continue;
      if (llvm::any_of(user->getRegions(), [](Region &r) {
            return llvm::any_of(r.getArguments(), [](BlockArgument bbArg) {
              return isa<MemRefType>(bbArg.getType());
            });
          }))
        continue;

      // For all other ops, we assume that we can directly replace the operand.
      // This may have to be revised in the future; e.g., there may be ops that
      // do not support non-identity layout maps.
      for (OpOperand &operand : user->getOpOperands()) {
        if ([[maybe_unused]] auto castOp =
                operand.get().getDefiningOp<UnrealizedConversionCastOp>()) {
          rewriter.modifyOpInPlace(
              user, [&]() { operand.set(conversion->getOperand(0)); });
        }
      }
    }
  }

  // Erase all unrealized_conversion_cast ops without uses.
  for (auto op : unrealizedConversions)
    if (op->getUses().empty())
      rewriter.eraseOp(op);
}

FailureOr<Value> memref::replaceWithIndependentOp(RewriterBase &rewriter,
                                                  memref::AllocaOp allocaOp,
                                                  ValueRange independencies) {
  auto replacement =
      memref::buildIndependentOp(rewriter, allocaOp, independencies);
  if (failed(replacement))
    return failure();
  replaceAndPropagateMemRefType(rewriter, allocaOp,
                                replacement->getDefiningOp());
  return replacement;
}

memref::AllocaOp memref::allocToAlloca(
    RewriterBase &rewriter, memref::AllocOp alloc,
    function_ref<bool(memref::AllocOp, memref::DeallocOp)> filter) {
  memref::DeallocOp dealloc = nullptr;
  for (Operation &candidate :
       llvm::make_range(alloc->getIterator(), alloc->getBlock()->end())) {
    dealloc = dyn_cast<memref::DeallocOp>(candidate);
    if (dealloc && dealloc.getMemref() == alloc.getMemref() &&
        (!filter || filter(alloc, dealloc))) {
      break;
    }
  }

  if (!dealloc)
    return nullptr;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(alloc);
  auto alloca = rewriter.replaceOpWithNewOp<memref::AllocaOp>(
      alloc, alloc.getMemref().getType(), alloc.getOperands());
  rewriter.eraseOp(dealloc);
  return alloca;
}
