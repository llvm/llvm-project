//===- BufferDeallocationOpInterface.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetOperations.h"

//===----------------------------------------------------------------------===//
// BufferDeallocationOpInterface
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bufferization {

#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.cpp.inc"

} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace bufferization;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value buildBoolValue(OpBuilder &builder, Location loc, bool value) {
  return builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(value));
}

static bool isMemref(Value v) { return v.getType().isa<BaseMemRefType>(); }

//===----------------------------------------------------------------------===//
// Ownership
//===----------------------------------------------------------------------===//

Ownership::Ownership(Value indicator)
    : indicator(indicator), state(State::Unique) {}

Ownership Ownership::getUnknown() {
  Ownership unknown;
  unknown.indicator = Value();
  unknown.state = State::Unknown;
  return unknown;
}
Ownership Ownership::getUnique(Value indicator) { return Ownership(indicator); }
Ownership Ownership::getUninitialized() { return Ownership(); }

bool Ownership::isUninitialized() const {
  return state == State::Uninitialized;
}
bool Ownership::isUnique() const { return state == State::Unique; }
bool Ownership::isUnknown() const { return state == State::Unknown; }

Value Ownership::getIndicator() const {
  assert(isUnique() && "must have unique ownership to get the indicator");
  return indicator;
}

Ownership Ownership::getCombined(Ownership other) const {
  if (other.isUninitialized())
    return *this;
  if (isUninitialized())
    return other;

  if (!isUnique() || !other.isUnique())
    return getUnknown();

  // Since we create a new constant i1 value for (almost) each use-site, we
  // should compare the actual value rather than just the SSA Value to avoid
  // unnecessary invalidations.
  if (isEqualConstantIntOrValue(indicator, other.indicator))
    return *this;

  // Return the join of the lattice if the indicator of both ownerships cannot
  // be merged.
  return getUnknown();
}

void Ownership::combine(Ownership other) { *this = getCombined(other); }

//===----------------------------------------------------------------------===//
// DeallocationState
//===----------------------------------------------------------------------===//

DeallocationState::DeallocationState(Operation *op) : liveness(op) {}

void DeallocationState::updateOwnership(Value memref, Ownership ownership,
                                        Block *block) {
  // In most cases we care about the block where the value is defined.
  if (block == nullptr)
    block = memref.getParentBlock();

  // Update ownership of current memref itself.
  ownershipMap[{memref, block}].combine(ownership);
}

void DeallocationState::resetOwnerships(ValueRange memrefs, Block *block) {
  for (Value val : memrefs)
    ownershipMap[{val, block}] = Ownership::getUninitialized();
}

Ownership DeallocationState::getOwnership(Value memref, Block *block) const {
  return ownershipMap.lookup({memref, block});
}

void DeallocationState::addMemrefToDeallocate(Value memref, Block *block) {
  memrefsToDeallocatePerBlock[block].push_back(memref);
}

void DeallocationState::dropMemrefToDeallocate(Value memref, Block *block) {
  llvm::erase_if(memrefsToDeallocatePerBlock[block],
                 [&](const auto &mr) { return mr == memref; });
}

void DeallocationState::getLiveMemrefsIn(Block *block,
                                         SmallVectorImpl<Value> &memrefs) {
  SmallVector<Value> liveMemrefs(
      llvm::make_filter_range(liveness.getLiveIn(block), isMemref));
  llvm::sort(liveMemrefs, ValueComparator());
  memrefs.append(liveMemrefs);
}

std::pair<Value, Value>
DeallocationState::getMemrefWithUniqueOwnership(OpBuilder &builder,
                                                Value memref) {
  auto iter = ownershipMap.find({memref, memref.getParentBlock()});
  assert(iter != ownershipMap.end() &&
         "Value must already have been registered in the ownership map");

  Ownership ownership = iter->second;
  if (ownership.isUnique())
    return {memref, ownership.getIndicator()};

  // Instead of inserting a clone operation we could also insert a dealloc
  // operation earlier in the block and use the updated ownerships returned by
  // the op for the retained values. Alternatively, we could insert code to
  // check aliasing at runtime and use this information to combine two unique
  // ownerships more intelligently to not end up with an 'Unknown' ownership in
  // the first place.
  auto cloneOp =
      builder.create<bufferization::CloneOp>(memref.getLoc(), memref);
  Value condition = buildBoolValue(builder, memref.getLoc(), true);
  Value newMemref = cloneOp.getResult();
  updateOwnership(newMemref, condition);
  memrefsToDeallocatePerBlock[newMemref.getParentBlock()].push_back(newMemref);
  return {newMemref, condition};
}

void DeallocationState::getMemrefsToRetain(
    Block *fromBlock, Block *toBlock, ValueRange destOperands,
    SmallVectorImpl<Value> &toRetain) const {
  for (Value operand : destOperands) {
    if (!isMemref(operand))
      continue;
    toRetain.push_back(operand);
  }

  SmallPtrSet<Value, 16> liveOut;
  for (auto val : liveness.getLiveOut(fromBlock))
    if (isMemref(val))
      liveOut.insert(val);

  if (toBlock)
    llvm::set_intersect(liveOut, liveness.getLiveIn(toBlock));

  // liveOut has non-deterministic order because it was constructed by iterating
  // over a hash-set.
  SmallVector<Value> retainedByLiveness(liveOut.begin(), liveOut.end());
  std::sort(retainedByLiveness.begin(), retainedByLiveness.end(),
            ValueComparator());
  toRetain.append(retainedByLiveness);
}

LogicalResult DeallocationState::getMemrefsAndConditionsToDeallocate(
    OpBuilder &builder, Location loc, Block *block,
    SmallVectorImpl<Value> &memrefs, SmallVectorImpl<Value> &conditions) const {

  for (auto [i, memref] :
       llvm::enumerate(memrefsToDeallocatePerBlock.lookup(block))) {
    Ownership ownership = ownershipMap.lookup({memref, block});
    if (!ownership.isUnique())
      return emitError(memref.getLoc(),
                       "MemRef value does not have valid ownership");

    // Simply cast unranked MemRefs to ranked memrefs with 0 dimensions such
    // that we can call extract_strided_metadata on it.
    if (auto unrankedMemRefTy = dyn_cast<UnrankedMemRefType>(memref.getType()))
      memref = builder.create<memref::ReinterpretCastOp>(
          loc, MemRefType::get({}, unrankedMemRefTy.getElementType()), memref,
          0, SmallVector<int64_t>{}, SmallVector<int64_t>{});

    // Use the `memref.extract_strided_metadata` operation to get the base
    // memref. This is needed because the same MemRef that was produced by the
    // alloc operation has to be passed to the dealloc operation. Passing
    // subviews, etc. to a dealloc operation is not allowed.
    memrefs.push_back(
        builder.create<memref::ExtractStridedMetadataOp>(loc, memref)
            .getResult(0));
    conditions.push_back(ownership.getIndicator());
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ValueComparator
//===----------------------------------------------------------------------===//

bool ValueComparator::operator()(const Value &lhs, const Value &rhs) const {
  if (lhs == rhs)
    return false;

  // Block arguments are less than results.
  bool lhsIsBBArg = lhs.isa<BlockArgument>();
  if (lhsIsBBArg != rhs.isa<BlockArgument>()) {
    return lhsIsBBArg;
  }

  Region *lhsRegion;
  Region *rhsRegion;
  if (lhsIsBBArg) {
    auto lhsBBArg = llvm::cast<BlockArgument>(lhs);
    auto rhsBBArg = llvm::cast<BlockArgument>(rhs);
    if (lhsBBArg.getArgNumber() != rhsBBArg.getArgNumber()) {
      return lhsBBArg.getArgNumber() < rhsBBArg.getArgNumber();
    }
    lhsRegion = lhsBBArg.getParentRegion();
    rhsRegion = rhsBBArg.getParentRegion();
    assert(lhsRegion != rhsRegion &&
           "lhsRegion == rhsRegion implies lhs == rhs");
  } else if (lhs.getDefiningOp() == rhs.getDefiningOp()) {
    return llvm::cast<OpResult>(lhs).getResultNumber() <
           llvm::cast<OpResult>(rhs).getResultNumber();
  } else {
    lhsRegion = lhs.getDefiningOp()->getParentRegion();
    rhsRegion = rhs.getDefiningOp()->getParentRegion();
    if (lhsRegion == rhsRegion) {
      return lhs.getDefiningOp()->isBeforeInBlock(rhs.getDefiningOp());
    }
  }

  // lhsRegion != rhsRegion, so if we look at their ancestor chain, they
  // - have different heights
  // - or there's a spot where their region numbers differ
  // - or their parent regions are the same and their parent ops are
  //   different.
  while (lhsRegion && rhsRegion) {
    if (lhsRegion->getRegionNumber() != rhsRegion->getRegionNumber()) {
      return lhsRegion->getRegionNumber() < rhsRegion->getRegionNumber();
    }
    if (lhsRegion->getParentRegion() == rhsRegion->getParentRegion()) {
      return lhsRegion->getParentOp()->isBeforeInBlock(
          rhsRegion->getParentOp());
    }
    lhsRegion = lhsRegion->getParentRegion();
    rhsRegion = rhsRegion->getParentRegion();
  }
  if (rhsRegion)
    return true;
  assert(lhsRegion && "this should only happen if lhs == rhs");
  return false;
}
