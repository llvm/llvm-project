//===- Value.cpp - MLIR Value Classes -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::detail;

/// If this value is the result of an Operation, return the operation that
/// defines it.
Operation *Value::getDefiningOp() const {
  if (auto result = dyn_cast<OpResult>())
    return result.getOwner();
  return nullptr;
}

Location Value::getLoc() const {
  if (auto *op = getDefiningOp())
    return op->getLoc();

  return cast<BlockArgument>().getLoc();
}

void Value::setLoc(Location loc) {
  if (auto *op = getDefiningOp())
    return op->setLoc(loc);

  return cast<BlockArgument>().setLoc(loc);
}

/// Return the Region in which this Value is defined.
Region *Value::getParentRegion() {
  if (auto *op = getDefiningOp())
    return op->getParentRegion();
  return cast<BlockArgument>().getOwner()->getParent();
}

/// Return the Block in which this Value is defined.
Block *Value::getParentBlock() {
  if (Operation *op = getDefiningOp())
    return op->getBlock();
  return cast<BlockArgument>().getOwner();
}

//===----------------------------------------------------------------------===//
// Value::UseLists
//===----------------------------------------------------------------------===//

/// Replace all uses of 'this' value with the new value, updating anything in
/// the IR that uses 'this' to use the other value instead except if the user is
/// listed in 'exceptions' .
void Value::replaceAllUsesExcept(
    Value newValue, const SmallPtrSetImpl<Operation *> &exceptions) const {
  for (OpOperand &use : llvm::make_early_inc_range(getUses())) {
    if (exceptions.count(use.getOwner()) == 0)
      use.set(newValue);
  }
}

/// Replace all uses of 'this' value with 'newValue', updating anything in the
/// IR that uses 'this' to use the other value instead except if the user is
/// 'exceptedUser'.
void Value::replaceAllUsesExcept(Value newValue,
                                 Operation *exceptedUser) const {
  for (OpOperand &use : llvm::make_early_inc_range(getUses())) {
    if (use.getOwner() != exceptedUser)
      use.set(newValue);
  }
}

/// Replace all uses of 'this' value with 'newValue' if the given callback
/// returns true.
void Value::replaceUsesWithIf(Value newValue,
                              function_ref<bool(OpOperand &)> shouldReplace) {
  for (OpOperand &use : llvm::make_early_inc_range(getUses()))
    if (shouldReplace(use))
      use.set(newValue);
}

/// Returns true if the value is used outside of the given block.
bool Value::isUsedOutsideOfBlock(Block *block) {
  return llvm::any_of(getUsers(), [block](Operation *user) {
    return user->getBlock() != block;
  });
}

//===----------------------------------------------------------------------===//
// OpResult
//===----------------------------------------------------------------------===//

/// Returns the parent operation of this trailing result.
Operation *OpResultImpl::getOwner() const {
  // We need to do some arithmetic to get the operation pointer. Results are
  // stored in reverse order before the operation, so move the trailing owner up
  // to the start of the array. A rough diagram of the memory layout is:
  //
  // | Out-of-Line results | Inline results | Operation |
  //
  // Given that the results are reverse order we use the result number to know
  // how far to jump to get to the operation. So if we are currently the 0th
  // result, the layout would be:
  //
  // | Inline result 0 | Operation
  //
  // ^-- To get the base address of the operation, we add the result count + 1.
  if (const auto *result = dyn_cast<InlineOpResult>(this)) {
    result += result->getResultNumber() + 1;
    return reinterpret_cast<Operation *>(const_cast<InlineOpResult *>(result));
  }

  // Out-of-line results are stored in an array just before the inline results.
  const OutOfLineOpResult *outOfLineIt = (const OutOfLineOpResult *)(this);
  outOfLineIt += (outOfLineIt->outOfLineIndex + 1);

  // Move the owner past the inline results to get to the operation.
  const auto *inlineIt = reinterpret_cast<const InlineOpResult *>(outOfLineIt);
  inlineIt += getMaxInlineResults();
  return reinterpret_cast<Operation *>(const_cast<InlineOpResult *>(inlineIt));
}

OpResultImpl *OpResultImpl::getNextResultAtOffset(intptr_t offset) {
  if (offset == 0)
    return this;
  // We need to do some arithmetic to get the next result given that results are
  // in reverse order, and that we need to account for the different types of
  // results. As a reminder, the rough diagram of the memory layout is:
  //
  // | Out-of-Line results | Inline results | Operation |
  //
  // So an example operation with two results would look something like:
  //
  // | Inline result 1 | Inline result 0 | Operation |
  //

  // Handle the case where this result is an inline result.
  OpResultImpl *result = this;
  if (auto *inlineResult = dyn_cast<InlineOpResult>(this)) {
    // Check to see how many results there are after this one before the start
    // of the out-of-line results. If the desired offset is less than the number
    // remaining, we can directly use the offset from the current result
    // pointer. The following diagrams highlight the two situations.
    //
    // | Out-of-Line results | Inline results | Operation |
    //                                    ^- Say we are here.
    //                           ^- If our destination is here, we can use the
    //                              offset directly.
    //
    intptr_t leftBeforeTrailing =
        getMaxInlineResults() - inlineResult->getResultNumber() - 1;
    if (leftBeforeTrailing >= offset)
      return inlineResult - offset;

    // Otherwise, adjust the current result pointer to the end (start in memory)
    // of the inline result array.
    //
    // | Out-of-Line results | Inline results | Operation |
    //                                    ^- Say we are here.
    //                  ^- If our destination is here, we need to first jump to
    //                     the end (start in memory) of the inline result array.
    //
    result = inlineResult - leftBeforeTrailing;
    offset -= leftBeforeTrailing;
  }

  // If we land here, the current result is an out-of-line result and we can
  // offset directly.
  return reinterpret_cast<OutOfLineOpResult *>(result) - offset;
}

/// Given a number of operation results, returns the number that need to be
/// stored inline.
unsigned OpResult::getNumInline(unsigned numResults) {
  return std::min(numResults, OpResultImpl::getMaxInlineResults());
}

/// Given a number of operation results, returns the number that need to be
/// stored as trailing.
unsigned OpResult::getNumTrailing(unsigned numResults) {
  // If we can pack all of the results, there is no need for additional storage.
  unsigned maxInline = OpResultImpl::getMaxInlineResults();
  return numResults <= maxInline ? 0 : numResults - maxInline;
}

//===----------------------------------------------------------------------===//
// BlockOperand
//===----------------------------------------------------------------------===//

/// Provide the use list that is attached to the given block.
IRObjectWithUseList<BlockOperand> *BlockOperand::getUseList(Block *value) {
  return value;
}

/// Return which operand this is in the operand list.
unsigned BlockOperand::getOperandNumber() {
  return this - &getOwner()->getBlockOperands()[0];
}

//===----------------------------------------------------------------------===//
// OpOperand
//===----------------------------------------------------------------------===//

/// Return which operand this is in the operand list.
unsigned OpOperand::getOperandNumber() {
  return this - &getOwner()->getOpOperands()[0];
}
