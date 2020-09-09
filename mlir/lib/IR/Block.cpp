//===- Block.cpp - MLIR Block Class ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// BlockArgument
//===----------------------------------------------------------------------===//

/// Returns the number of this argument.
unsigned BlockArgument::getArgNumber() const {
  // Arguments are not stored in place, so we have to find it within the list.
  auto argList = getOwner()->getArguments();
  return std::distance(argList.begin(), llvm::find(argList, *this));
}

//===----------------------------------------------------------------------===//
// Block
//===----------------------------------------------------------------------===//

Block::~Block() {
  assert(!verifyOpOrder() && "Expected valid operation ordering.");
  clear();
  for (BlockArgument arg : arguments)
    arg.destroy();
}

Region *Block::getParent() const { return parentValidOpOrderPair.getPointer(); }

/// Returns the closest surrounding operation that contains this block or
/// nullptr if this block is unlinked.
Operation *Block::getParentOp() {
  return getParent() ? getParent()->getParentOp() : nullptr;
}

/// Return if this block is the entry block in the parent region.
bool Block::isEntryBlock() { return this == &getParent()->front(); }

/// Insert this block (which must not already be in a region) right before the
/// specified block.
void Block::insertBefore(Block *block) {
  assert(!getParent() && "already inserted into a block!");
  assert(block->getParent() && "cannot insert before a block without a parent");
  block->getParent()->getBlocks().insert(block->getIterator(), this);
}

/// Unlink this block from its current region and insert it right before the
/// specific block.
void Block::moveBefore(Block *block) {
  assert(block->getParent() && "cannot insert before a block without a parent");
  block->getParent()->getBlocks().splice(
      block->getIterator(), getParent()->getBlocks(), getIterator());
}

/// Unlink this Block from its parent Region and delete it.
void Block::erase() {
  assert(getParent() && "Block has no parent");
  getParent()->getBlocks().erase(this);
}

/// Returns 'op' if 'op' lies in this block, or otherwise finds the
/// ancestor operation of 'op' that lies in this block. Returns nullptr if
/// the latter fails.
Operation *Block::findAncestorOpInBlock(Operation &op) {
  // Traverse up the operation hierarchy starting from the owner of operand to
  // find the ancestor operation that resides in the block of 'forOp'.
  auto *currOp = &op;
  while (currOp->getBlock() != this) {
    currOp = currOp->getParentOp();
    if (!currOp)
      return nullptr;
  }
  return currOp;
}

/// This drops all operand uses from operations within this block, which is
/// an essential step in breaking cyclic dependences between references when
/// they are to be deleted.
void Block::dropAllReferences() {
  for (Operation &i : *this)
    i.dropAllReferences();
}

void Block::dropAllDefinedValueUses() {
  for (auto arg : getArguments())
    arg.dropAllUses();
  for (auto &op : *this)
    op.dropAllDefinedValueUses();
  dropAllUses();
}

/// Returns true if the ordering of the child operations is valid, false
/// otherwise.
bool Block::isOpOrderValid() { return parentValidOpOrderPair.getInt(); }

/// Invalidates the current ordering of operations.
void Block::invalidateOpOrder() {
  // Validate the current ordering.
  assert(!verifyOpOrder());
  parentValidOpOrderPair.setInt(false);
}

/// Verifies the current ordering of child operations. Returns false if the
/// order is valid, true otherwise.
bool Block::verifyOpOrder() {
  // The order is already known to be invalid.
  if (!isOpOrderValid())
    return false;
  // The order is valid if there are less than 2 operations.
  if (operations.empty() || std::next(operations.begin()) == operations.end())
    return false;

  Operation *prev = nullptr;
  for (auto &i : *this) {
    // The previous operation must have a smaller order index than the next as
    // it appears earlier in the list.
    if (prev && prev->orderIndex != Operation::kInvalidOrderIdx &&
        prev->orderIndex >= i.orderIndex)
      return true;
    prev = &i;
  }
  return false;
}

/// Recomputes the ordering of child operations within the block.
void Block::recomputeOpOrder() {
  parentValidOpOrderPair.setInt(true);

  unsigned orderIndex = 0;
  for (auto &op : *this)
    op.orderIndex = (orderIndex += Operation::kOrderStride);
}

//===----------------------------------------------------------------------===//
// Argument list management.
//===----------------------------------------------------------------------===//

/// Return a range containing the types of the arguments for this block.
auto Block::getArgumentTypes() -> ValueTypeRange<BlockArgListType> {
  return ValueTypeRange<BlockArgListType>(getArguments());
}

BlockArgument Block::addArgument(Type type) {
  BlockArgument arg = BlockArgument::create(type, this);
  arguments.push_back(arg);
  return arg;
}

/// Add one argument to the argument list for each type specified in the list.
auto Block::addArguments(TypeRange types) -> iterator_range<args_iterator> {
  size_t initialSize = arguments.size();
  arguments.reserve(initialSize + types.size());
  for (auto type : types)
    addArgument(type);
  return {arguments.data() + initialSize, arguments.data() + arguments.size()};
}

BlockArgument Block::insertArgument(unsigned index, Type type) {
  auto arg = BlockArgument::create(type, this);
  assert(index <= arguments.size());
  arguments.insert(arguments.begin() + index, arg);
  return arg;
}

void Block::eraseArgument(unsigned index) {
  assert(index < arguments.size());
  arguments[index].destroy();
  arguments.erase(arguments.begin() + index);
}

/// Insert one value to the given position of the argument list. The existing
/// arguments are shifted. The block is expected not to have predecessors.
BlockArgument Block::insertArgument(args_iterator it, Type type) {
  assert(llvm::empty(getPredecessors()) &&
         "cannot insert arguments to blocks with predecessors");

  // Use the args_iterator (on the BlockArgListType) to compute the insertion
  // iterator in the underlying argument storage.
  size_t distance = std::distance(args_begin(), it);
  auto arg = BlockArgument::create(type, this);
  arguments.insert(std::next(arguments.begin(), distance), arg);
  return arg;
}

//===----------------------------------------------------------------------===//
// Terminator management
//===----------------------------------------------------------------------===//

/// Get the terminator operation of this block. This function asserts that
/// the block has a valid terminator operation.
Operation *Block::getTerminator() {
  assert(!empty() && !back().isKnownNonTerminator());
  return &back();
}

// Indexed successor access.
unsigned Block::getNumSuccessors() {
  return empty() ? 0 : back().getNumSuccessors();
}

Block *Block::getSuccessor(unsigned i) {
  assert(i < getNumSuccessors());
  return getTerminator()->getSuccessor(i);
}

/// If this block has exactly one predecessor, return it.  Otherwise, return
/// null.
///
/// Note that multiple edges from a single block (e.g. if you have a cond
/// branch with the same block as the true/false destinations) is not
/// considered to be a single predecessor.
Block *Block::getSinglePredecessor() {
  auto it = pred_begin();
  if (it == pred_end())
    return nullptr;
  auto *firstPred = *it;
  ++it;
  return it == pred_end() ? firstPred : nullptr;
}

/// If this block has a unique predecessor, i.e., all incoming edges originate
/// from one block, return it. Otherwise, return null.
Block *Block::getUniquePredecessor() {
  auto it = pred_begin(), e = pred_end();
  if (it == e)
    return nullptr;

  // Check for any conflicting predecessors.
  auto *firstPred = *it;
  for (++it; it != e; ++it)
    if (*it != firstPred)
      return nullptr;
  return firstPred;
}

//===----------------------------------------------------------------------===//
// Other
//===----------------------------------------------------------------------===//

/// Split the block into two blocks before the specified operation or
/// iterator.
///
/// Note that all operations BEFORE the specified iterator stay as part of
/// the original basic block, and the rest of the operations in the original
/// block are moved to the new block, including the old terminator.  The
/// original block is left without a terminator.
///
/// The newly formed Block is returned, and the specified iterator is
/// invalidated.
Block *Block::splitBlock(iterator splitBefore) {
  // Start by creating a new basic block, and insert it immediate after this
  // one in the containing region.
  auto newBB = new Block();
  getParent()->getBlocks().insert(std::next(Region::iterator(this)), newBB);

  // Move all of the operations from the split point to the end of the region
  // into the new block.
  newBB->getOperations().splice(newBB->end(), getOperations(), splitBefore,
                                end());
  return newBB;
}

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

Block *PredecessorIterator::unwrap(BlockOperand &value) {
  return value.getOwner()->getBlock();
}

/// Get the successor number in the predecessor terminator.
unsigned PredecessorIterator::getSuccessorIndex() const {
  return I->getOperandNumber();
}

//===----------------------------------------------------------------------===//
// SuccessorRange
//===----------------------------------------------------------------------===//

SuccessorRange::SuccessorRange(Block *block) : SuccessorRange(nullptr, 0) {
  if (Operation *term = block->getTerminator())
    if ((count = term->getNumSuccessors()))
      base = term->getBlockOperands().data();
}

SuccessorRange::SuccessorRange(Operation *term) : SuccessorRange(nullptr, 0) {
  if ((count = term->getNumSuccessors()))
    base = term->getBlockOperands().data();
}

//===----------------------------------------------------------------------===//
// BlockRange
//===----------------------------------------------------------------------===//

BlockRange::BlockRange(ArrayRef<Block *> blocks) : BlockRange(nullptr, 0) {
  if ((count = blocks.size()))
    base = blocks.data();
}

BlockRange::BlockRange(SuccessorRange successors)
    : BlockRange(successors.begin().getBase(), successors.size()) {}

/// See `llvm::detail::indexed_accessor_range_base` for details.
BlockRange::OwnerT BlockRange::offset_base(OwnerT object, ptrdiff_t index) {
  if (auto *operand = object.dyn_cast<BlockOperand *>())
    return {operand + index};
  return {object.dyn_cast<Block *const *>() + index};
}

/// See `llvm::detail::indexed_accessor_range_base` for details.
Block *BlockRange::dereference_iterator(OwnerT object, ptrdiff_t index) {
  if (const auto *operand = object.dyn_cast<BlockOperand *>())
    return operand[index].get();
  return object.dyn_cast<Block *const *>()[index];
}
