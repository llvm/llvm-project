//===- Region.cpp - MLIR Region Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Region.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
using namespace mlir;

Region::Region(Operation *container) : container(container) {}

Region::~Region() {
  // Operations may have cyclic references, which need to be dropped before we
  // can start deleting them.
  dropAllReferences();
}

/// Return the context this region is inserted in. The region must have a valid
/// parent container.
MLIRContext *Region::getContext() {
  assert(container && "region is not attached to a container");
  return container->getContext();
}

/// Return a location for this region. This is the location attached to the
/// parent container. The region must have a valid parent container.
Location Region::getLoc() {
  assert(container && "region is not attached to a container");
  return container->getLoc();
}

auto Region::getArgumentTypes() -> ValueTypeRange<BlockArgListType> {
  return ValueTypeRange<BlockArgListType>(getArguments());
}

iterator_range<Region::args_iterator>
Region::addArguments(TypeRange types, ArrayRef<Location> locs) {
  return front().addArguments(types, locs);
}

Region *Region::getParentRegion() {
  assert(container && "region is not attached to a container");
  return container->getParentRegion();
}

bool Region::isProperAncestor(Region *other) {
  if (this == other)
    return false;

  while ((other = other->getParentRegion())) {
    if (this == other)
      return true;
  }
  return false;
}

/// Return the number of this region in the parent operation.
unsigned Region::getRegionNumber() {
  // Regions are always stored consecutively, so use pointer subtraction to
  // figure out what number this is.
  return this - &getParentOp()->getRegions()[0];
}

/// Clone the internal blocks from this region into `dest`. Any
/// cloned blocks are appended to the back of dest.
void Region::cloneInto(Region *dest, IRMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  cloneInto(dest, dest->end(), mapper);
}

/// Clone this region into 'dest' before the given position in 'dest'.
void Region::cloneInto(Region *dest, Region::iterator destPos,
                       IRMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  assert(this != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (empty())
    return;

  // The below clone implementation takes special care to be read only for the
  // sake of multi threading. That essentially means not adding any uses to any
  // of the blocks or operation results contained within this region as that
  // would lead to a write in their use-def list. This is unavoidable for
  // 'Value's from outside the region however, in which case it is not read
  // only. Using the BlockAndValueMapper it is possible to remap such 'Value's
  // to ones owned by the calling thread however, making it read only once
  // again.

  // First clone all the blocks and block arguments and map them, but don't yet
  // clone the operations, as they may otherwise add a use to a block that has
  // not yet been mapped
  for (Block &block : *this) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    dest->getBlocks().insert(destPos, newBlock);
  }

  auto newBlocksRange =
      llvm::make_range(Region::iterator(mapper.lookup(&front())), destPos);

  // Now follow up with creating the operations, but don't yet clone their
  // regions, nor set their operands. Setting the successors is safe as all have
  // already been mapped. We are essentially just creating the operation results
  // to be able to map them.
  // Cloning the operands and region as well would lead to uses of operations
  // not yet mapped.
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto zippedBlocks : llvm::zip(*this, newBlocksRange)) {
    Block &sourceBlock = std::get<0>(zippedBlocks);
    Block &clonedBlock = std::get<1>(zippedBlocks);
    // Clone and remap the operations within this block.
    for (Operation &op : sourceBlock)
      clonedBlock.push_back(op.clone(mapper, cloneOptions));
  }

  // Finally now that all operation results have been mapped, set the operands
  // and clone the regions.
  SmallVector<Value> operands;
  for (auto zippedBlocks : llvm::zip(*this, newBlocksRange)) {
    for (auto ops :
         llvm::zip(std::get<0>(zippedBlocks), std::get<1>(zippedBlocks))) {
      Operation &source = std::get<0>(ops);
      Operation &clone = std::get<1>(ops);

      operands.resize(source.getNumOperands());
      llvm::transform(
          source.getOperands(), operands.begin(),
          [&](Value operand) { return mapper.lookupOrDefault(operand); });
      clone.setOperands(operands);

      for (auto regions : llvm::zip(source.getRegions(), clone.getRegions()))
        std::get<0>(regions).cloneInto(&std::get<1>(regions), mapper);
    }
  }
}

/// Returns 'block' if 'block' lies in this region, or otherwise finds the
/// ancestor of 'block' that lies in this region. Returns nullptr if the latter
/// fails.
Block *Region::findAncestorBlockInRegion(Block &block) {
  Block *currBlock = &block;
  while (currBlock->getParent() != this) {
    Operation *parentOp = currBlock->getParentOp();
    if (!parentOp || !parentOp->getBlock())
      return nullptr;
    currBlock = parentOp->getBlock();
  }
  return currBlock;
}

/// Returns 'op' if 'op' lies in this region, or otherwise finds the
/// ancestor of 'op' that lies in this region. Returns nullptr if the
/// latter fails.
Operation *Region::findAncestorOpInRegion(Operation &op) {
  Operation *curOp = &op;
  while (Region *opRegion = curOp->getParentRegion()) {
    if (opRegion == this)
      return curOp;

    curOp = opRegion->getParentOp();
    if (!curOp)
      return nullptr;
  }
  return nullptr;
}

void Region::dropAllReferences() {
  for (Block &b : *this)
    b.dropAllReferences();
}

Region *llvm::ilist_traits<::mlir::Block>::getParentRegion() {
  size_t offset(
      size_t(&((Region *)nullptr->*Region::getSublistAccess(nullptr))));
  iplist<Block> *anchor(static_cast<iplist<Block> *>(this));
  return reinterpret_cast<Region *>(reinterpret_cast<char *>(anchor) - offset);
}

/// This is a trait method invoked when a basic block is added to a region.
/// We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::addNodeToList(Block *block) {
  assert(!block->getParent() && "already in a region!");
  block->parentValidOpOrderPair.setPointer(getParentRegion());
}

/// This is a trait method invoked when an operation is removed from a
/// region.  We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::Block>::removeNodeFromList(Block *block) {
  assert(block->getParent() && "not already in a region!");
  block->parentValidOpOrderPair.setPointer(nullptr);
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::Block>::transferNodesFromList(
    ilist_traits<Block> &otherList, block_iterator first, block_iterator last) {
  // If we are transferring operations within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getParentRegion();
  if (curParent == otherList.getParentRegion())
    return;

  // Update the 'parent' member of each Block.
  for (; first != last; ++first)
    first->parentValidOpOrderPair.setPointer(curParent);
}

//===----------------------------------------------------------------------===//
// Region::OpIterator
//===----------------------------------------------------------------------===//

Region::OpIterator::OpIterator(Region *region, bool end)
    : region(region), block(end ? region->end() : region->begin()) {
  if (!region->empty())
    skipOverBlocksWithNoOps();
}

Region::OpIterator &Region::OpIterator::operator++() {
  // We increment over operations, if we reach the last use then move to next
  // block.
  if (operation != block->end())
    ++operation;
  if (operation == block->end()) {
    ++block;
    skipOverBlocksWithNoOps();
  }
  return *this;
}

void Region::OpIterator::skipOverBlocksWithNoOps() {
  while (block != region->end() && block->empty())
    ++block;

  // If we are at the last block, then set the operation to first operation of
  // next block (sentinel value used for end).
  if (block == region->end())
    operation = {};
  else
    operation = block->begin();
}

//===----------------------------------------------------------------------===//
// RegionRange
//===----------------------------------------------------------------------===//

RegionRange::RegionRange(MutableArrayRef<Region> regions)
    : RegionRange(regions.data(), regions.size()) {}
RegionRange::RegionRange(ArrayRef<std::unique_ptr<Region>> regions)
    : RegionRange(regions.data(), regions.size()) {}
RegionRange::RegionRange(ArrayRef<Region *> regions)
    : RegionRange(const_cast<Region **>(regions.data()), regions.size()) {}

/// See `llvm::detail::indexed_accessor_range_base` for details.
RegionRange::OwnerT RegionRange::offset_base(const OwnerT &owner,
                                             ptrdiff_t index) {
  if (auto *region = owner.dyn_cast<const std::unique_ptr<Region> *>())
    return region + index;
  if (auto **region = owner.dyn_cast<Region **>())
    return region + index;
  return &owner.get<Region *>()[index];
}
/// See `llvm::detail::indexed_accessor_range_base` for details.
Region *RegionRange::dereference_iterator(const OwnerT &owner,
                                          ptrdiff_t index) {
  if (auto *region = owner.dyn_cast<const std::unique_ptr<Region> *>())
    return region[index].get();
  if (auto **region = owner.dyn_cast<Region **>())
    return region[index];
  return &owner.get<Region *>()[index];
}
