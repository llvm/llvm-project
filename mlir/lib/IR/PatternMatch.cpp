//===- PatternMatch.cpp - Base classes for pattern match ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/RegionKindInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PatternBenefit
//===----------------------------------------------------------------------===//

PatternBenefit::PatternBenefit(unsigned benefit) : representation(benefit) {
  assert(representation == benefit && benefit != ImpossibleToMatchSentinel &&
         "This pattern match benefit is too large to represent");
}

unsigned short PatternBenefit::getBenefit() const {
  assert(!isImpossibleToMatch() && "Pattern doesn't match");
  return representation;
}

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// OperationName Root Constructors

Pattern::Pattern(StringRef rootName, PatternBenefit benefit,
                 MLIRContext *context, ArrayRef<StringRef> generatedNames)
    : Pattern(OperationName(rootName, context).getAsOpaquePointer(),
              RootKind::OperationName, generatedNames, benefit, context) {}

//===----------------------------------------------------------------------===//
// MatchAnyOpTypeTag Root Constructors

Pattern::Pattern(MatchAnyOpTypeTag tag, PatternBenefit benefit,
                 MLIRContext *context, ArrayRef<StringRef> generatedNames)
    : Pattern(nullptr, RootKind::Any, generatedNames, benefit, context) {}

//===----------------------------------------------------------------------===//
// MatchInterfaceOpTypeTag Root Constructors

Pattern::Pattern(MatchInterfaceOpTypeTag tag, TypeID interfaceID,
                 PatternBenefit benefit, MLIRContext *context,
                 ArrayRef<StringRef> generatedNames)
    : Pattern(interfaceID.getAsOpaquePointer(), RootKind::InterfaceID,
              generatedNames, benefit, context) {}

//===----------------------------------------------------------------------===//
// MatchTraitOpTypeTag Root Constructors

Pattern::Pattern(MatchTraitOpTypeTag tag, TypeID traitID,
                 PatternBenefit benefit, MLIRContext *context,
                 ArrayRef<StringRef> generatedNames)
    : Pattern(traitID.getAsOpaquePointer(), RootKind::TraitID, generatedNames,
              benefit, context) {}

//===----------------------------------------------------------------------===//
// General Constructors

Pattern::Pattern(const void *rootValue, RootKind rootKind,
                 ArrayRef<StringRef> generatedNames, PatternBenefit benefit,
                 MLIRContext *context)
    : rootValue(rootValue), rootKind(rootKind), benefit(benefit),
      contextAndHasBoundedRecursion(context, false) {
  if (generatedNames.empty())
    return;
  generatedOps.reserve(generatedNames.size());
  std::transform(generatedNames.begin(), generatedNames.end(),
                 std::back_inserter(generatedOps), [context](StringRef name) {
                   return OperationName(name, context);
                 });
}

//===----------------------------------------------------------------------===//
// RewritePattern
//===----------------------------------------------------------------------===//

void RewritePattern::rewrite(Operation *op, PatternRewriter &rewriter) const {
  llvm_unreachable("need to implement either matchAndRewrite or one of the "
                   "rewrite functions!");
}

LogicalResult RewritePattern::match(Operation *op) const {
  llvm_unreachable("need to implement either match or matchAndRewrite!");
}

/// Out-of-line vtable anchor.
void RewritePattern::anchor() {}

//===----------------------------------------------------------------------===//
// RewriterBase
//===----------------------------------------------------------------------===//

bool RewriterBase::Listener::classof(const OpBuilder::Listener *base) {
  return base->getKind() == OpBuilder::ListenerBase::Kind::RewriterBaseListener;
}

RewriterBase::~RewriterBase() {
  // Out of line to provide a vtable anchor for the class.
}

/// This method replaces the results of the operation with the specified list of
/// values. The number of provided values must match the number of results of
/// the operation. The replaced op is erased.
void RewriterBase::replaceOp(Operation *op, ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");

  // Notify the listener that we're about to replace this op.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(op, newValues);

  // Replace all result uses. Also notifies the listener of modifications.
  replaceAllOpUsesWith(op, newValues);

  // Erase op and notify listener.
  eraseOp(op);
}

/// This method replaces the results of the operation with the specified new op
/// (replacement). The number of results of the two operations must match. The
/// replaced op is erased.
void RewriterBase::replaceOp(Operation *op, Operation *newOp) {
  assert(op && newOp && "expected non-null op");
  assert(op->getNumResults() == newOp->getNumResults() &&
         "ops have different number of results");

  // Notify the listener that we're about to replace this op.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(op, newOp);

  // Replace all result uses. Also notifies the listener of modifications.
  replaceAllOpUsesWith(op, newOp->getResults());

  // Erase op and notify listener.
  eraseOp(op);
}

/// This method erases an operation that is known to have no uses. The uses of
/// the given operation *must* be known to be dead.
void RewriterBase::eraseOp(Operation *op) {
  assert(op->use_empty() && "expected 'op' to have no uses");
  auto *rewriteListener = dyn_cast_if_present<Listener>(listener);

  // Fast path: If no listener is attached, the op can be dropped in one go.
  if (!rewriteListener) {
    op->erase();
    return;
  }

  // Helper function that erases a single op.
  auto eraseSingleOp = [&](Operation *op) {
#ifndef NDEBUG
    // All nested ops should have been erased already.
    assert(
        llvm::all_of(op->getRegions(), [&](Region &r) { return r.empty(); }) &&
        "expected empty regions");
    // All users should have been erased already if the op is in a region with
    // SSA dominance.
    if (!op->use_empty() && op->getParentOp())
      assert(mayBeGraphRegion(*op->getParentRegion()) &&
             "expected that op has no uses");
#endif // NDEBUG
    rewriteListener->notifyOperationErased(op);

    // Explicitly drop all uses in case the op is in a graph region.
    op->dropAllUses();
    op->erase();
  };

  // Nested ops must be erased one-by-one, so that listeners have a consistent
  // view of the IR every time a notification is triggered. Users must be
  // erased before definitions. I.e., post-order, reverse dominance.
  std::function<void(Operation *)> eraseTree = [&](Operation *op) {
    // Erase nested ops.
    for (Region &r : llvm::reverse(op->getRegions())) {
      // Erase all blocks in the right order. Successors should be erased
      // before predecessors because successor blocks may use values defined
      // in predecessor blocks. A post-order traversal of blocks within a
      // region visits successors before predecessors. Repeat the traversal
      // until the region is empty. (The block graph could be disconnected.)
      while (!r.empty()) {
        SmallVector<Block *> erasedBlocks;
        // Some blocks may have invalid successor, use a set including nullptr
        // to avoid null pointer.
        llvm::SmallPtrSet<Block *, 4> visited{nullptr};
        for (Block *b : llvm::post_order_ext(&r.front(), visited)) {
          // Visit ops in reverse order.
          for (Operation &op :
               llvm::make_early_inc_range(ReverseIterator::makeIterable(*b)))
            eraseTree(&op);
          // Do not erase the block immediately. This is not supprted by the
          // post_order iterator.
          erasedBlocks.push_back(b);
        }
        for (Block *b : erasedBlocks) {
          // Explicitly drop all uses in case there is a cycle in the block
          // graph.
          for (BlockArgument bbArg : b->getArguments())
            bbArg.dropAllUses();
          b->dropAllUses();
          eraseBlock(b);
        }
      }
    }
    // Then erase the enclosing op.
    eraseSingleOp(op);
  };

  eraseTree(op);
}

void RewriterBase::eraseBlock(Block *block) {
  assert(block->use_empty() && "expected 'block' to have no uses");

  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    eraseOp(&op);
  }

  // Notify the listener that the block is about to be removed.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyBlockErased(block);

  block->erase();
}

void RewriterBase::finalizeOpModification(Operation *op) {
  // Notify the listener that the operation was modified.
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationModified(op);
}

void RewriterBase::replaceUsesWithIf(Value from, Value to,
                                     function_ref<bool(OpOperand &)> functor,
                                     bool *allUsesReplaced) {
  bool allReplaced = true;
  for (OpOperand &operand : llvm::make_early_inc_range(from.getUses())) {
    bool replace = functor(operand);
    if (replace)
      modifyOpInPlace(operand.getOwner(), [&]() { operand.set(to); });
    allReplaced &= replace;
  }
  if (allUsesReplaced)
    *allUsesReplaced = allReplaced;
}

void RewriterBase::replaceUsesWithIf(ValueRange from, ValueRange to,
                                     function_ref<bool(OpOperand &)> functor,
                                     bool *allUsesReplaced) {
  assert(from.size() == to.size() && "incorrect number of replacements");
  bool allReplaced = true;
  for (auto it : llvm::zip_equal(from, to)) {
    bool r;
    replaceUsesWithIf(std::get<0>(it), std::get<1>(it), functor,
                      /*allUsesReplaced=*/&r);
    allReplaced &= r;
  }
  if (allUsesReplaced)
    *allUsesReplaced = allReplaced;
}

void RewriterBase::inlineBlockBefore(Block *source, Block *dest,
                                     Block::iterator before,
                                     ValueRange argValues) {
  assert(argValues.size() == source->getNumArguments() &&
         "incorrect # of argument replacement values");

  // The source block will be deleted, so it should not have any users (i.e.,
  // there should be no predecessors).
  assert(source->hasNoPredecessors() &&
         "expected 'source' to have no predecessors");

  if (dest->end() != before) {
    // The source block will be inserted in the middle of the dest block, so
    // the source block should have no successors. Otherwise, the remainder of
    // the dest block would be unreachable.
    assert(source->hasNoSuccessors() &&
           "expected 'source' to have no successors");
  } else {
    // The source block will be inserted at the end of the dest block, so the
    // dest block should have no successors. Otherwise, the inserted operations
    // will be unreachable.
    assert(dest->hasNoSuccessors() && "expected 'dest' to have no successors");
  }

  // Replace all of the successor arguments with the provided values.
  for (auto it : llvm::zip(source->getArguments(), argValues))
    replaceAllUsesWith(std::get<0>(it), std::get<1>(it));

  // Move operations from the source block to the dest block and erase the
  // source block.
  if (!listener) {
    // Fast path: If no listener is attached, move all operations at once.
    dest->getOperations().splice(before, source->getOperations());
  } else {
    while (!source->empty())
      moveOpBefore(&source->front(), dest, before);
  }

  // Erase the source block.
  assert(source->empty() && "expected 'source' to be empty");
  eraseBlock(source);
}

void RewriterBase::inlineBlockBefore(Block *source, Operation *op,
                                     ValueRange argValues) {
  inlineBlockBefore(source, op->getBlock(), op->getIterator(), argValues);
}

void RewriterBase::mergeBlocks(Block *source, Block *dest,
                               ValueRange argValues) {
  inlineBlockBefore(source, dest, dest->end(), argValues);
}

/// Split the operations starting at "before" (inclusive) out of the given
/// block into a new block, and return it.
Block *RewriterBase::splitBlock(Block *block, Block::iterator before) {
  // Fast path: If no listener is attached, split the block directly.
  if (!listener)
    return block->splitBlock(before);

  // `createBlock` sets the insertion point at the beginning of the new block.
  InsertionGuard g(*this);
  Block *newBlock =
      createBlock(block->getParent(), std::next(block->getIterator()));

  // If `before` points to end of the block, no ops should be moved.
  if (before == block->end())
    return newBlock;

  // Move ops one-by-one from the end of `block` to the beginning of `newBlock`.
  // Stop when the operation pointed to by `before` has been moved.
  while (before->getBlock() != newBlock)
    moveOpBefore(&block->back(), newBlock, newBlock->begin());

  return newBlock;
}

/// Move the blocks that belong to "region" before the given position in
/// another region.  The two regions must be different.  The caller is in
/// charge to update create the operation transferring the control flow to the
/// region and pass it the correct block arguments.
void RewriterBase::inlineRegionBefore(Region &region, Region &parent,
                                      Region::iterator before) {
  // Fast path: If no listener is attached, move all blocks at once.
  if (!listener) {
    parent.getBlocks().splice(before, region.getBlocks());
    return;
  }

  // Move blocks from the beginning of the region one-by-one.
  while (!region.empty())
    moveBlockBefore(&region.front(), &parent, before);
}
void RewriterBase::inlineRegionBefore(Region &region, Block *before) {
  inlineRegionBefore(region, *before->getParent(), before->getIterator());
}

void RewriterBase::moveBlockBefore(Block *block, Block *anotherBlock) {
  moveBlockBefore(block, anotherBlock->getParent(),
                  anotherBlock->getIterator());
}

void RewriterBase::moveBlockBefore(Block *block, Region *region,
                                   Region::iterator iterator) {
  Region *currentRegion = block->getParent();
  Region::iterator nextIterator = std::next(block->getIterator());
  block->moveBefore(region, iterator);
  if (listener)
    listener->notifyBlockInserted(block, /*previous=*/currentRegion,
                                  /*previousIt=*/nextIterator);
}

void RewriterBase::moveOpBefore(Operation *op, Operation *existingOp) {
  moveOpBefore(op, existingOp->getBlock(), existingOp->getIterator());
}

void RewriterBase::moveOpBefore(Operation *op, Block *block,
                                Block::iterator iterator) {
  Block *currentBlock = op->getBlock();
  Block::iterator nextIterator = std::next(op->getIterator());
  op->moveBefore(block, iterator);
  if (listener)
    listener->notifyOperationInserted(
        op, /*previous=*/InsertPoint(currentBlock, nextIterator));
}

void RewriterBase::moveOpAfter(Operation *op, Operation *existingOp) {
  moveOpAfter(op, existingOp->getBlock(), existingOp->getIterator());
}

void RewriterBase::moveOpAfter(Operation *op, Block *block,
                               Block::iterator iterator) {
  assert(iterator != block->end() && "cannot move after end of block");
  moveOpBefore(op, block, std::next(iterator));
}
