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

Value DeallocationState::materializeMemRefOwnership(
    const DeallocationOptions &options, OpBuilder &builder, Value memref,
    Block *block) {
  // NOTE: Starts at the operation defining `memref` and performs a DFS along
  // the reverse def/use chain until MemRef values with 'Unique' ownership are
  // found. For the operation being currently processed:
  // * if none of the operands have the same allocated pointer (i.e., originate
  //   from the same allocation), a new memref was allocated and thus the
  //   operation should have the allocate side-effect defined on that result
  //   value and thus the correct unique ownership is pre-populated by the
  //   ownership pass (unless an interface implementation is incorrect). Note
  //   that this is problematic for operations of unregistered dialects because
  //   the allocation side-effect cannot be represented in the assembly format.
  // * if exactly one operand has the same allocated pointer, this returnes the
  //   ownership of exactly that operand
  // * if multiple operands match the allocated pointer of the result, the
  //   ownership indicators of all of them always have to evaluate to the same
  //   value because no dealloc operations may be present and because of the
  //   rules they are passed to nested regions and successor blocks.  This could
  //   be verified at runtime by inserting `cf.assert` operations, but would
  //   require O(|operands|^2) additional operations to check and is thus not
  //   implemented yet (would need to insert a library function to avoid
  //   code-size explosion which would make the deallocation pass a module pass)
  auto ipSave = builder.saveInsertionPoint();
  SmallVector<Value> worklist;
  worklist.push_back(memref);

  while (!worklist.empty()) {
    Value curr = worklist.back();

    // If the value already has unique ownership, we don't have to process it
    // anymore.
    Ownership ownership = getOwnership(curr, block);
    if (ownership.isUnique()) {
      worklist.pop_back();
      continue;
    }

    // Check if all operands of MemRef type have unique ownership.
    Operation *defOp = curr.getDefiningOp();
    assert(defOp &&
           "the ownership-based deallocation pass should be written in a way "
           "that pre-populates ownership for block arguments");

    bool allKnown = true;
    for (Value val : llvm::make_filter_range(defOp->getOperands(), isMemref)) {
      Ownership ownership = getOwnership(val, block);
      if (ownership.isUnique())
        continue;

      worklist.push_back(val);
      allKnown = false;
    }

    // If all MemRef operands have unique ownership, we can check if the op
    // implements the BufferDeallocationOpInterface and call that or, otherwise,
    // we call the generic implementation manually here.
    if (allKnown) {
      builder.setInsertionPointAfter(defOp);
      if (auto deallocInterface =
              dyn_cast<BufferDeallocationOpInterface>(defOp);
          deallocInterface && curr.getParentBlock() == block)
        ownership = deallocInterface.materializeUniqueOwnershipForMemref(
            *this, options, builder, curr);
      else
        ownership = deallocation_impl::defaultComputeMemRefOwnership(
            options, *this, builder, curr, block);

      // Ownership is already 'Unknown', so we need to override instead of
      // joining.
      resetOwnerships(curr, block);
      updateOwnership(curr, ownership, block);
    }
  }

  builder.restoreInsertionPoint(ipSave);
  return getOwnership(memref, block).getIndicator();
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

//===----------------------------------------------------------------------===//
// Implementation utilities
//===----------------------------------------------------------------------===//

FailureOr<Operation *> deallocation_impl::insertDeallocOpForReturnLike(
    DeallocationState &state, Operation *op, ValueRange operands,
    SmallVectorImpl<Value> &updatedOperandOwnerships) {
  assert(op->hasTrait<OpTrait::IsTerminator>() && "must be a terminator");
  assert(!op->hasSuccessors() && "must not have any successors");
  // Collect the values to deallocate and retain and use them to create the
  // dealloc operation.
  OpBuilder builder(op);
  Block *block = op->getBlock();
  SmallVector<Value> memrefs, conditions, toRetain;
  if (failed(state.getMemrefsAndConditionsToDeallocate(
          builder, op->getLoc(), block, memrefs, conditions)))
    return failure();

  state.getMemrefsToRetain(block, /*toBlock=*/nullptr, operands, toRetain);
  if (memrefs.empty() && toRetain.empty())
    return op;

  auto deallocOp = builder.create<bufferization::DeallocOp>(
      op->getLoc(), memrefs, conditions, toRetain);

  // We want to replace the current ownership of the retained values with the
  // result values of the dealloc operation as they are always unique.
  state.resetOwnerships(deallocOp.getRetained(), block);
  for (auto [retained, ownership] :
       llvm::zip(deallocOp.getRetained(), deallocOp.getUpdatedConditions()))
    state.updateOwnership(retained, ownership, block);

  unsigned numMemrefOperands = llvm::count_if(operands, isMemref);
  auto newOperandOwnerships =
      deallocOp.getUpdatedConditions().take_front(numMemrefOperands);
  updatedOperandOwnerships.append(newOperandOwnerships.begin(),
                                  newOperandOwnerships.end());

  return op;
}

Value deallocation_impl::defaultComputeMemRefOwnership(
    const DeallocationOptions &options, DeallocationState &state,
    OpBuilder &builder, Value memref, Block *block) {
  Operation *defOp = memref.getDefiningOp();
  SmallVector<Value> operands(
      llvm::make_filter_range(defOp->getOperands(), isMemref));
  Value resultPtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
      defOp->getLoc(), memref);
  Value ownership = state.getOwnership(operands.front(), block).getIndicator();

  for (Value val : ArrayRef(operands).drop_front()) {
    Value operandPtr = builder.create<memref::ExtractAlignedPointerAsIndexOp>(
        defOp->getLoc(), val);
    Value isSameBuffer = builder.create<arith::CmpIOp>(
        defOp->getLoc(), arith::CmpIPredicate::eq, resultPtr, operandPtr);
    Value newOwnership = state.getOwnership(val, block).getIndicator();
    ownership = builder.create<arith::SelectOp>(defOp->getLoc(), isSameBuffer,
                                                newOwnership, ownership);
  }
  return ownership;
}
