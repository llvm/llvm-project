//===- CFGToSCF.h - Control Flow Graph to Structured Control Flow *- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This code is an implementation of:
// Helge Bahmann, Nico Reissmann, Magnus Jahre, and Jan Christian Meyer. 2015.
// Perfect Reconstructability of Control Flow from Demand Dependence Graphs. ACM
// Trans. Archit. Code Optim. 11, 4, Article 66 (January 2015), 25 pages.
// https://doi.org/10.1145/2693261
//
// It defines an algorithm to translate any control flow graph with a single
// entry and single exit block into structured control flow operations
// consisting of regions of do-while loops and operations conditionally
// dispatching to one out of multiple regions before continuing after the
// operation. This includes control flow graphs containing irreducible
// control flow.
//
// The implementation here additionally supports the transformation on
// regions with multiple exit blocks. This is implemented by first
// transforming all occurrences of return-like operations to branch to a
// single exit block containing an instance of that return-like operation.
// If there are multiple kinds of return-like operations, multiple exit
// blocks are created. In that case the transformation leaves behind a
// conditional control flow graph operation that dispatches to the given regions
// terminating with different kinds of return-like operations each.
//
// If the function only contains a single kind of return-like operations,
// it is guaranteed that all control flow graph ops will be lifted to structured
// control flow, and that no more control flow graph ops remain after the
// operation.
//
// The algorithm to lift CFGs consists of two transformations applied after each
// other on any single-entry, single-exit region:
// 1) Lifting cycles to structured control flow loops
// 2) Lifting conditional branches to structured control flow branches
// These are then applied recursively on any new single-entry single-exit
// regions created by the transformation until no more CFG operations remain.
//
// The first part of cycle lifting is to detect any cycles in the CFG.
// This is done using an algorithm for iterating over SCCs. Every SCC
// representing a cycle is then transformed into a structured loop with a single
// entry block and a single latch containing the only back edge to the entry
// block and the only edge to an exit block outside the loop. Rerouting control
// flow to create single entry and exit blocks is achieved via a multiplexer
// construct that can be visualized as follows:
//                         +-----+ +-----+   +-----+
//                         | bb0 | | bb1 |...| bbN |
//                         +--+--+ +--+--+   +-+---+
//                            |       |        |
//                            |       v        |
//                            |  +------+      |
//                            | ++      ++<----+
//                            | | Region |
//                            +>|        |<----+
//                              ++      ++     |
//                               +------+------+
//
// The above transforms to:
//                         +-----+ +-----+   +-----+
//                         | bb0 | | bb1 |...| bbN |
//                         +-----+ +--|--+   ++----+
//                              |     v       |
//                              +->+-----+<---+
//                                 | bbM |<-------+
//                                 +---+-+        |
//                             +---+   | +----+   |
//                             |       v      |   |
//                             |   +------+   |   |
//                             |  ++      ++<-+   |
//                             +->| Region |      |
//                                ++      ++      |
//                                 +------+-------+
//
// bbM in the above is the multiplexer block, and any block previously branching
// to an entry block of the region are redirected to it. This includes any
// branches from within the region. Using a block argument, bbM then dispatches
// to the correct entry block of the region dependent on the predecessor.
//
// A similar transformation is done to create the latch block with the single
// back edge and loop exit edge.
//
// The above form has the advantage that bbM now acts as the loop header
// of the loop body. After the transformation on the latch, this results in a
// structured loop that can then be lifted to structured control flow. The
// conditional branches created in bbM are later lifted to conditional
// branches.
//
// Lifting conditional branches is done by analyzing the *first* conditional
// branch encountered in the entry region. The algorithm then identifies
// all blocks that are dominated by a specific control flow edge and
// the region where control flow continues:
//                                 +-----+
//                           +-----+ bb0 +----+
//                           v     +-----+    v
//                Region 1 +-+-+    ...     +-+-+ Region n
//                         +---+            +---+
//                          ...              ...
//                           |                |
//                           |      +---+     |
//                           +---->++   ++<---+
//                                 |     |
//                                 ++   ++ Region T
//                                  +---+
// Every region following bb0 consists of 0 or more blocks that eventually
// branch to Region T. If there are multiple entry blocks into Region T, a
// single entry block is created using a multiplexer block as shown above.
// Region 1 to Region n are then lifted together with the conditional control
// flow operation terminating bb0 into a structured conditional operation
// followed by the operations of the entry block of Region T.
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CFGToSCF.h"

#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

/// Returns the mutable operand range used to transfer operands from `block` to
/// its successor with the given index. The returned range being mutable allows
/// us to modify the operands being transferred.
static MutableOperandRange
getMutableSuccessorOperands(Block *block, unsigned successorIndex) {
  auto branchOpInterface = cast<BranchOpInterface>(block->getTerminator());
  SuccessorOperands succOps =
      branchOpInterface.getSuccessorOperands(successorIndex);
  return succOps.getMutableForwardedOperands();
}

/// Return the operand range used to transfer operands from `block` to its
/// successor with the given index.
static OperandRange getSuccessorOperands(Block *block,
                                         unsigned successorIndex) {
  return getMutableSuccessorOperands(block, successorIndex);
}

/// Appends all the block arguments from `other` to the block arguments of
/// `block`, copying their types and locations.
static void addBlockArgumentsFromOther(Block *block, Block *other) {
  for (BlockArgument arg : other->getArguments())
    block->addArgument(arg.getType(), arg.getLoc());
}

namespace {

/// Class representing an edge in the CFG. Consists of a from-block, a successor
/// and corresponding successor operands passed to the block arguments of the
/// successor.
class Edge {
  Block *fromBlock;
  unsigned successorIndex;

public:
  /// Constructs a new edge from `fromBlock` to the successor corresponding to
  /// `successorIndex`.
  Edge(Block *fromBlock, unsigned int successorIndex)
      : fromBlock(fromBlock), successorIndex(successorIndex) {}

  /// Returns the from-block.
  Block *getFromBlock() const { return fromBlock; }

  /// Returns the successor of the edge.
  Block *getSuccessor() const {
    return fromBlock->getSuccessor(successorIndex);
  }

  /// Sets the successor of the edge, adjusting the terminator in the
  /// from-block.
  void setSuccessor(Block *block) const {
    fromBlock->getTerminator()->setSuccessor(block, successorIndex);
  }

  /// Returns the arguments of this edge that are passed to the block arguments
  /// of the successor.
  MutableOperandRange getMutableSuccessorOperands() const {
    return ::getMutableSuccessorOperands(fromBlock, successorIndex);
  }

  /// Returns the arguments of this edge that are passed to the block arguments
  /// of the successor.
  OperandRange getSuccessorOperands() const {
    return ::getSuccessorOperands(fromBlock, successorIndex);
  }
};

/// Structure containing the entry, exit and back edges of a cycle. A cycle is a
/// generalization of a loop that may have multiple entry edges. See also
/// https://llvm.org/docs/CycleTerminology.html.
struct CycleEdges {
  /// All edges from a block outside the cycle to a block inside the cycle.
  /// The targets of these edges are entry blocks.
  SmallVector<Edge> entryEdges;
  /// All edges from a block inside the cycle to a block outside the cycle.
  SmallVector<Edge> exitEdges;
  /// All edges from a block inside the cycle to an entry block.
  SmallVector<Edge> backEdges;
};

/// Class used to orchestrate creation of so-called edge multiplexers.
/// This class creates a new basic block and routes all inputs edges
/// to this basic block before branching to their original target.
/// The purpose of this transformation is to create single-entry,
/// single-exit regions.
class EdgeMultiplexer {
public:
  /// Creates a new edge multiplexer capable of redirecting all edges to one of
  /// the `entryBlocks`. This creates the multiplexer basic block with
  /// appropriate block arguments after the first entry block. `extraArgs`
  /// contains the types of possible extra block arguments passed to the
  /// multiplexer block that are added to the successor operands of every
  /// outgoing edge.
  ///
  /// NOTE: This does not yet redirect edges to branch to the
  /// multiplexer block nor code dispatching from the multiplexer code
  /// to the original successors.
  /// See `redirectEdge` and `createSwitch`.
  static EdgeMultiplexer create(Location loc, ArrayRef<Block *> entryBlocks,
                                function_ref<Value(unsigned)> getSwitchValue,
                                function_ref<Value(Type)> getUndefValue,
                                TypeRange extraArgs = {}) {
    assert(!entryBlocks.empty() && "Require at least one entry block");

    auto *multiplexerBlock = new Block;
    multiplexerBlock->insertAfter(entryBlocks.front());

    // To implement the multiplexer block, we have to add the block arguments of
    // every distinct successor block to the multiplexer block. When redirecting
    // edges, block arguments designated for blocks that aren't branched to will
    // be assigned the `getUndefValue`. The amount of block arguments and their
    // offset is saved in the map for `redirectEdge` to transform the edges.
    llvm::SmallMapVector<Block *, unsigned, 4> blockArgMapping;
    for (Block *entryBlock : entryBlocks) {
      auto [iter, inserted] = blockArgMapping.insert(
          {entryBlock, multiplexerBlock->getNumArguments()});
      if (inserted)
        addBlockArgumentsFromOther(multiplexerBlock, entryBlock);
    }

    // If we have more than one successor, we have to additionally add a
    // discriminator value, denoting which successor to jump to.
    // When redirecting edges, an appropriate value will be passed using
    // `getSwitchValue`.
    Value discriminator;
    if (blockArgMapping.size() > 1)
      discriminator =
          multiplexerBlock->addArgument(getSwitchValue(0).getType(), loc);

    multiplexerBlock->addArguments(
        extraArgs, SmallVector<Location>(extraArgs.size(), loc));

    return EdgeMultiplexer(multiplexerBlock, getSwitchValue, getUndefValue,
                           std::move(blockArgMapping), discriminator);
  }

  /// Returns the created multiplexer block.
  Block *getMultiplexerBlock() const { return multiplexerBlock; }

  /// Redirects `edge` to branch to the multiplexer block before continuing to
  /// its original target. The edges successor must have originally been part
  /// of the entry blocks array passed to the `create` function. `extraArgs`
  /// must be used to pass along any additional values corresponding to
  /// `extraArgs` in `create`.
  void redirectEdge(Edge edge, ValueRange extraArgs = {}) const {
    const auto *result = blockArgMapping.find(edge.getSuccessor());
    assert(result != blockArgMapping.end() &&
           "Edge was not originally passed to `create` method.");

    MutableOperandRange successorOperands = edge.getMutableSuccessorOperands();

    // Extra arguments are always appended at the end of the block arguments.
    unsigned extraArgsBeginIndex =
        multiplexerBlock->getNumArguments() - extraArgs.size();
    // If a discriminator exists, it is right before the extra arguments.
    std::optional<unsigned> discriminatorIndex =
        discriminator ? extraArgsBeginIndex - 1 : std::optional<unsigned>{};

    SmallVector<Value> newSuccOperands(multiplexerBlock->getNumArguments());
    for (BlockArgument argument : multiplexerBlock->getArguments()) {
      unsigned index = argument.getArgNumber();
      if (index >= result->second &&
          index < result->second + edge.getSuccessor()->getNumArguments()) {
        // Original block arguments to the entry block.
        newSuccOperands[index] =
            successorOperands[index - result->second].get();
        continue;
      }

      // Discriminator value if it exists.
      if (index == discriminatorIndex) {
        newSuccOperands[index] =
            getSwitchValue(result - blockArgMapping.begin());
        continue;
      }

      // Followed by the extra arguments.
      if (index >= extraArgsBeginIndex) {
        newSuccOperands[index] = extraArgs[index - extraArgsBeginIndex];
        continue;
      }

      // Otherwise undef values for any unused block arguments used by other
      // entry blocks.
      newSuccOperands[index] = getUndefValue(argument.getType());
    }

    edge.setSuccessor(multiplexerBlock);
    successorOperands.assign(newSuccOperands);
  }

  /// Creates a switch op using `builder` which dispatches to the original
  /// successors of the edges passed to `create` minus the ones in `excluded`.
  /// The builder's insertion point has to be in a block dominated by the
  /// multiplexer block. All edges to the multiplexer block must have already
  /// been redirected using `redirectEdge`.
  void createSwitch(
      Location loc, OpBuilder &builder, CFGToSCFInterface &interface,
      const SmallPtrSetImpl<Block *> &excluded = SmallPtrSet<Block *, 1>{}) {
    // We create the switch by creating a case for all entries and then
    // splitting of the last entry as a default case.

    SmallVector<ValueRange> caseArguments;
    SmallVector<unsigned> caseValues;
    SmallVector<Block *> caseDestinations;
    for (auto &&[index, pair] : llvm::enumerate(blockArgMapping)) {
      auto &&[succ, offset] = pair;
      if (excluded.contains(succ))
        continue;

      caseValues.push_back(index);
      caseArguments.push_back(multiplexerBlock->getArguments().slice(
          offset, succ->getNumArguments()));
      caseDestinations.push_back(succ);
    }

    // If we don't have a discriminator due to only having one entry we have to
    // create a dummy flag for the switch.
    Value realDiscriminator = discriminator;
    if (!realDiscriminator || caseArguments.size() == 1)
      realDiscriminator = getSwitchValue(0);

    caseValues.pop_back();
    Block *defaultDest = caseDestinations.pop_back_val();
    ValueRange defaultArgs = caseArguments.pop_back_val();

    assert(!builder.getInsertionBlock()->hasNoPredecessors() &&
           "Edges need to be redirected prior to creating switch.");
    interface.createCFGSwitchOp(loc, builder, realDiscriminator, caseValues,
                                caseDestinations, caseArguments, defaultDest,
                                defaultArgs);
  }

private:
  /// Newly created multiplexer block.
  Block *multiplexerBlock;
  /// Callback used to create a constant suitable as flag for
  /// the interfaces `createCFGSwitchOp`.
  function_ref<Value(unsigned)> getSwitchValue;
  /// Callback used to create undefined values of a given type.
  function_ref<Value(Type)> getUndefValue;

  /// Mapping of the block arguments of an entry block to the corresponding
  /// block arguments in the multiplexer block. Block arguments of an entry
  /// block are simply appended ot the multiplexer block. This map simply
  /// contains the offset to the range in the multiplexer block.
  llvm::SmallMapVector<Block *, unsigned, 4> blockArgMapping;
  /// Discriminator value used in the multiplexer block to dispatch to the
  /// correct entry block. Null value if not required due to only having one
  /// entry block.
  Value discriminator;

  EdgeMultiplexer(Block *multiplexerBlock,
                  function_ref<Value(unsigned)> getSwitchValue,
                  function_ref<Value(Type)> getUndefValue,
                  llvm::SmallMapVector<Block *, unsigned, 4> &&entries,
                  Value dispatchFlag)
      : multiplexerBlock(multiplexerBlock), getSwitchValue(getSwitchValue),
        getUndefValue(getUndefValue), blockArgMapping(std::move(entries)),
        discriminator(dispatchFlag) {}
};

/// Alternative implementation of DenseMapInfo<Operation*> using the operation
/// equivalence infrastructure to check whether two 'return-like' operations are
/// equivalent in the context of this transformation. This means that both
/// operations are of the same kind, have the same amount of operands and types
/// and the same attributes and properties. The operands themselves don't have
/// to be equivalent.
struct ReturnLikeOpEquivalence : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::ignoreHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }

  static bool isEqual(const Operation *lhs, const Operation *rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhs), const_cast<Operation *>(rhs),
        OperationEquivalence::ignoreValueEquivalence, nullptr,
        OperationEquivalence::IgnoreLocations);
  }
};

/// Utility-class for transforming a region to only have one single block for
/// every return-like operation.
class ReturnLikeExitCombiner {
public:
  ReturnLikeExitCombiner(Region &topLevelRegion, CFGToSCFInterface &interface)
      : topLevelRegion(topLevelRegion), interface(interface) {}

  /// Transforms `returnLikeOp` to a branch to the only block in the
  /// region with an instance of `returnLikeOp`s kind.
  void combineExit(Operation *returnLikeOp,
                   function_ref<Value(unsigned)> getSwitchValue) {
    auto [iter, inserted] =
        returnLikeToCombinedExit.insert({returnLikeOp, nullptr});
    if (!inserted && iter->first == returnLikeOp)
      return;

    Block *exitBlock = iter->second;
    if (inserted) {
      exitBlock = new Block;
      iter->second = exitBlock;
      topLevelRegion.push_back(exitBlock);
      exitBlock->addArguments(
          returnLikeOp->getOperandTypes(),
          SmallVector<Location>(returnLikeOp->getNumOperands(),
                                returnLikeOp->getLoc()));
    }

    auto builder = OpBuilder::atBlockTerminator(returnLikeOp->getBlock());
    interface.createSingleDestinationBranch(returnLikeOp->getLoc(), builder,
                                            getSwitchValue(0), exitBlock,
                                            returnLikeOp->getOperands());

    if (!inserted) {
      returnLikeOp->erase();
      return;
    }

    returnLikeOp->moveBefore(exitBlock, exitBlock->end());
    returnLikeOp->setOperands(exitBlock->getArguments());
  }

private:
  /// Mapping of return-like operation to block. All return-like operations
  /// of the same kind with the same attributes, properties and types are seen
  /// as equivalent. First occurrence seen is kept in the map.
  llvm::SmallDenseMap<Operation *, Block *, 4, ReturnLikeOpEquivalence>
      returnLikeToCombinedExit;
  Region &topLevelRegion;
  CFGToSCFInterface &interface;
};

} // namespace

/// Returns a range of all edges from `block` to each of its successors.
static auto successorEdges(Block *block) {
  return llvm::map_range(llvm::seq(block->getNumSuccessors()),
                         [=](unsigned index) { return Edge(block, index); });
}

/// Calculates entry, exit and back edges of the given cycle.
static CycleEdges
calculateCycleEdges(const llvm::SmallSetVector<Block *, 4> &cycles) {
  CycleEdges result;
  SmallPtrSet<Block *, 8> entryBlocks;

  // First identify all exit and entry edges by checking whether any successors
  // or predecessors are from outside the cycles.
  for (Block *block : cycles) {
    for (auto pred = block->pred_begin(); pred != block->pred_end(); pred++) {
      if (cycles.contains(*pred))
        continue;

      result.entryEdges.emplace_back(*pred, pred.getSuccessorIndex());
      entryBlocks.insert(block);
    }

    for (auto &&[succIndex, succ] : llvm::enumerate(block->getSuccessors())) {
      if (cycles.contains(succ))
        continue;

      result.exitEdges.emplace_back(block, succIndex);
    }
  }

  // With the entry blocks identified, find all the back edges.
  for (Block *block : cycles) {
    for (auto &&[succIndex, succ] : llvm::enumerate(block->getSuccessors())) {
      if (!entryBlocks.contains(succ))
        continue;

      result.backEdges.emplace_back(block, succIndex);
    }
  }

  return result;
}

/// Creates a single entry block out of multiple entry edges using an edge
/// multiplexer and returns it.
static EdgeMultiplexer
createSingleEntryBlock(Location loc, ArrayRef<Edge> entryEdges,
                       function_ref<Value(unsigned)> getSwitchValue,
                       function_ref<Value(Type)> getUndefValue,
                       CFGToSCFInterface &interface) {
  auto result = EdgeMultiplexer::create(
      loc, llvm::map_to_vector(entryEdges, std::mem_fn(&Edge::getSuccessor)),
      getSwitchValue, getUndefValue);

  // Redirect the edges prior to creating the switch op.
  // We guarantee that predecessors are up to date.
  for (Edge edge : entryEdges)
    result.redirectEdge(edge);

  auto builder = OpBuilder::atBlockBegin(result.getMultiplexerBlock());
  result.createSwitch(loc, builder, interface);

  return result;
}

namespace {
/// Special loop properties of a structured loop.
/// A structured loop is a loop satisfying all of the following:
/// * Has at most one entry, one exit and one back edge.
/// * The back edge originates from the same block as the exit edge.
struct StructuredLoopProperties {
  /// Block containing both the single exit edge and the single back edge.
  Block *latch;
  /// Loop condition of type equal to a value returned by `getSwitchValue`.
  Value condition;
  /// Exit block which is the only successor of the loop.
  Block *exitBlock;
};
} // namespace

/// Transforms a loop into a structured loop with only a single back edge and
/// exiting edge, originating from the same block.
static FailureOr<StructuredLoopProperties> createSingleExitingLatch(
    Location loc, ArrayRef<Edge> backEdges, ArrayRef<Edge> exitEdges,
    function_ref<Value(unsigned)> getSwitchValue,
    function_ref<Value(Type)> getUndefValue, CFGToSCFInterface &interface,
    ReturnLikeExitCombiner &exitCombiner) {
  assert(llvm::all_equal(
             llvm::map_range(backEdges, std::mem_fn(&Edge::getSuccessor))) &&
         "All repetition edges must lead to the single loop header");

  // First create the multiplexer block, which will be our latch, for all back
  // edges and exit edges. We pass an additional argument to the multiplexer
  // block which indicates whether the latch was reached from what was
  // originally a back edge or an exit block.
  // This is later used to branch using the new only back edge.
  SmallVector<Block *> successors;
  llvm::append_range(
      successors, llvm::map_range(backEdges, std::mem_fn(&Edge::getSuccessor)));
  llvm::append_range(
      successors, llvm::map_range(exitEdges, std::mem_fn(&Edge::getSuccessor)));
  auto multiplexer =
      EdgeMultiplexer::create(loc, successors, getSwitchValue, getUndefValue,
                              /*extraArgs=*/getSwitchValue(0).getType());

  auto *latchBlock = multiplexer.getMultiplexerBlock();

  // Create a separate exit block that comes right after the latch.
  auto *exitBlock = new Block;
  exitBlock->insertAfter(latchBlock);

  // Since this is a loop, all back edges point to the same loop header.
  Block *loopHeader = backEdges.front().getSuccessor();

  // Redirect the edges prior to creating the switch op.
  // We guarantee that predecessors are up to date.

  // Redirecting back edges with `shouldRepeat` as 1.
  for (Edge backEdge : backEdges)
    multiplexer.redirectEdge(backEdge, /*extraArgs=*/getSwitchValue(1));

  // Redirecting exits edges with `shouldRepeat` as 0.
  for (Edge exitEdge : exitEdges)
    multiplexer.redirectEdge(exitEdge, /*extraArgs=*/getSwitchValue(0));

  // Create the new only back edge to the loop header. Branch to the
  // exit block otherwise.
  Value shouldRepeat = latchBlock->getArguments().back();
  {
    auto builder = OpBuilder::atBlockBegin(latchBlock);
    interface.createConditionalBranch(
        builder.getUnknownLoc(), builder, shouldRepeat, loopHeader,
        latchBlock->getArguments().take_front(loopHeader->getNumArguments()),
        /*falseDest=*/exitBlock,
        /*falseArgs=*/{});
  }

  {
    auto builder = OpBuilder::atBlockBegin(exitBlock);
    if (!exitEdges.empty()) {
      // Create the switch dispatching to what were originally the multiple exit
      // blocks. The loop header has to explicitly be excluded in the below
      // switch as we would otherwise be creating a new loop again. All back
      // edges leading to the loop header have already been handled in the
      // switch above. The remaining edges can only jump to blocks outside the
      // loop.

      SmallPtrSet<Block *, 1> excluded = {loopHeader};
      multiplexer.createSwitch(loc, builder, interface, excluded);
    } else {
      // A loop without an exit edge is a statically known infinite loop.
      // Since structured control flow ops are not terminator ops, the caller
      // has to create a fitting return-like unreachable terminator operation.
      FailureOr<Operation *> terminator = interface.createUnreachableTerminator(
          loc, builder, *latchBlock->getParent());
      if (failed(terminator))
        return failure();
      // Transform the just created transform operation in the case that an
      // occurrence of it existed in input IR.
      exitCombiner.combineExit(*terminator, getSwitchValue);
    }
  }

  return StructuredLoopProperties{latchBlock, /*condition=*/shouldRepeat,
                                  exitBlock};
}

/// Transforms a structured loop into a loop in reduce form.
///
/// Reduce form is defined as a structured loop where:
/// (0) No values defined within the loop body are used outside the loop body.
/// (1) The block arguments and successor operands of the exit block are equal
///     to the block arguments of the loop header and the successor operands
///     of the back edge.
///
/// This is required for many structured control flow ops as they tend
/// to not have separate "loop result arguments" and "loop iteration arguments"
/// at the end of the block. Rather, the "loop iteration arguments" from the
/// last iteration are the result of the loop.
///
/// Note that the requirement of (0) is shared with LCSSA form in LLVM. However,
/// due to this being a structured loop instead of a general loop, we do not
/// require complicated dominance algorithms nor SSA updating making this
/// implementation easier than creating a generic LCSSA transformation pass.
static SmallVector<Value>
transformToReduceLoop(Block *loopHeader, Block *exitBlock,
                      const llvm::SmallSetVector<Block *, 4> &loopBlocks,
                      function_ref<Value(Type)> getUndefValue,
                      DominanceInfo &dominanceInfo) {
  Block *latch = exitBlock->getSinglePredecessor();
  assert(latch &&
         "Exit block must have only latch as predecessor at this point");
  assert(exitBlock->getNumArguments() == 0 &&
         "Exit block mustn't have any block arguments at this point");

  unsigned loopHeaderIndex = 0;
  unsigned exitBlockIndex = 1;
  if (latch->getSuccessor(loopHeaderIndex) != loopHeader)
    std::swap(loopHeaderIndex, exitBlockIndex);

  assert(latch->getSuccessor(loopHeaderIndex) == loopHeader);
  assert(latch->getSuccessor(exitBlockIndex) == exitBlock);

  MutableOperandRange exitBlockSuccessorOperands =
      getMutableSuccessorOperands(latch, exitBlockIndex);
  // Save the values as a vector, not a `MutableOperandRange` as the latter gets
  // invalidated when mutating the operands through a different
  // `MutableOperandRange` of the same operation.
  SmallVector<Value> loopHeaderSuccessorOperands =
      llvm::to_vector(getSuccessorOperands(latch, loopHeaderIndex));

  // Add all values used in the next iteration to the exit block. Replace
  // any uses that are outside the loop with the newly created exit block.
  for (Value arg : loopHeaderSuccessorOperands) {
    BlockArgument exitArg = exitBlock->addArgument(arg.getType(), arg.getLoc());
    exitBlockSuccessorOperands.append(arg);
    arg.replaceUsesWithIf(exitArg, [&](OpOperand &use) {
      return !loopBlocks.contains(use.getOwner()->getBlock());
    });
  }

  // Loop below might add block arguments to the latch and loop header.
  // Save the block arguments prior to the loop to not process these.
  SmallVector<BlockArgument> latchBlockArgumentsPrior =
      llvm::to_vector(latch->getArguments());
  SmallVector<BlockArgument> loopHeaderArgumentsPrior =
      llvm::to_vector(loopHeader->getArguments());

  // Go over all values defined within the loop body. If any of them are used
  // outside the loop body, create a block argument on the exit block and loop
  // header and replace the outside uses with the exit block argument.
  // The loop header block argument is added to satisfy requirement (1) in the
  // reduce form condition.
  for (Block *loopBlock : loopBlocks) {
    // Cache dominance queries for loopBlock.
    // There are likely to be many duplicate queries as there can be many value
    // definitions within a block.
    llvm::SmallDenseMap<Block *, bool> dominanceCache;
    // Returns true if `loopBlock` dominates `block`.
    auto loopBlockDominates = [&](Block *block) {
      auto [iter, inserted] = dominanceCache.insert({block, false});
      if (!inserted)
        return iter->second;
      iter->second = dominanceInfo.dominates(loopBlock, block);
      return iter->second;
    };

    auto checkValue = [&](Value value) {
      Value blockArgument;
      for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
        // Go through all the parent blocks and find the one part of the region
        // of the loop. If the block is part of the loop, then the value does
        // not escape the loop through this use.
        Block *currBlock = use.getOwner()->getBlock();
        while (currBlock && currBlock->getParent() != loopHeader->getParent())
          currBlock = currBlock->getParentOp()->getBlock();
        if (loopBlocks.contains(currBlock))
          continue;

        // Block argument is only created the first time it is required.
        if (!blockArgument) {
          blockArgument =
              exitBlock->addArgument(value.getType(), value.getLoc());
          loopHeader->addArgument(value.getType(), value.getLoc());

          // `value` might be defined in a block that does not dominate `latch`
          // but previously dominated an exit block with a use.
          // In this case, add a block argument to the latch and go through all
          // predecessors. If the value dominates the predecessor, pass the
          // value as a successor operand, otherwise pass undef.
          // The above is unnecessary if the value is a block argument of the
          // latch or if `value` dominates all predecessors.
          Value argument = value;
          if (value.getParentBlock() != latch &&
              llvm::any_of(latch->getPredecessors(), [&](Block *pred) {
                return !loopBlockDominates(pred);
              })) {
            argument = latch->addArgument(value.getType(), value.getLoc());
            for (auto iter = latch->pred_begin(); iter != latch->pred_end();
                 ++iter) {
              Value succOperand = value;
              if (!loopBlockDominates(*iter))
                succOperand = getUndefValue(value.getType());

              getMutableSuccessorOperands(*iter, iter.getSuccessorIndex())
                  .append(succOperand);
            }
          }

          loopHeaderSuccessorOperands.push_back(argument);
          for (Edge edge : successorEdges(latch))
            edge.getMutableSuccessorOperands().append(argument);
        }

        use.set(blockArgument);
      }
    };

    if (loopBlock == latch)
      llvm::for_each(latchBlockArgumentsPrior, checkValue);
    else if (loopBlock == loopHeader)
      llvm::for_each(loopHeaderArgumentsPrior, checkValue);
    else
      llvm::for_each(loopBlock->getArguments(), checkValue);

    for (Operation &op : *loopBlock)
      llvm::for_each(op.getResults(), checkValue);
  }

  // New block arguments may have been added to the loop header.
  // Adjust the entry edges to pass undef values to these.
  for (auto iter = loopHeader->pred_begin(); iter != loopHeader->pred_end();
       ++iter) {
    // Latch successor arguments have already been handled.
    if (*iter == latch)
      continue;

    MutableOperandRange succOps =
        getMutableSuccessorOperands(*iter, iter.getSuccessorIndex());
    succOps.append(llvm::map_to_vector(
        loopHeader->getArguments().drop_front(succOps.size()),
        [&](BlockArgument arg) { return getUndefValue(arg.getType()); }));
  }

  return loopHeaderSuccessorOperands;
}

/// Transforms all outer-most cycles in the region with the region entry
/// `regionEntry` into structured loops. Returns the entry blocks of any newly
/// created regions potentially requiring further transformations.
static FailureOr<SmallVector<Block *>> transformCyclesToSCFLoops(
    Block *regionEntry, function_ref<Value(unsigned)> getSwitchValue,
    function_ref<Value(Type)> getUndefValue, CFGToSCFInterface &interface,
    DominanceInfo &dominanceInfo, ReturnLikeExitCombiner &exitCombiner) {
  SmallVector<Block *> newSubRegions;
  auto scc = llvm::scc_begin(regionEntry);
  while (!scc.isAtEnd()) {
    if (!scc.hasCycle()) {
      ++scc;
      continue;
    }

    // Save the set and increment the SCC iterator early to avoid our
    // modifications breaking the SCC iterator.
    llvm::SmallSetVector<Block *, 4> cycleBlockSet(scc->begin(), scc->end());
    ++scc;

    CycleEdges edges = calculateCycleEdges(cycleBlockSet);
    Block *loopHeader = edges.entryEdges.front().getSuccessor();
    // First turn the cycle into a loop by creating a single entry block if
    // needed.
    if (edges.entryEdges.size() > 1) {
      SmallVector<Edge> edgesToEntryBlocks;
      llvm::append_range(edgesToEntryBlocks, edges.entryEdges);
      llvm::append_range(edgesToEntryBlocks, edges.backEdges);

      EdgeMultiplexer multiplexer = createSingleEntryBlock(
          loopHeader->getTerminator()->getLoc(), edgesToEntryBlocks,
          getSwitchValue, getUndefValue, interface);

      loopHeader = multiplexer.getMultiplexerBlock();
    }
    cycleBlockSet.insert(loopHeader);

    // Then turn it into a structured loop by creating a single latch.
    FailureOr<StructuredLoopProperties> loopProperties =
        createSingleExitingLatch(
            edges.backEdges.front().getFromBlock()->getTerminator()->getLoc(),
            edges.backEdges, edges.exitEdges, getSwitchValue, getUndefValue,
            interface, exitCombiner);
    if (failed(loopProperties))
      return failure();

    Block *latchBlock = loopProperties->latch;
    Block *exitBlock = loopProperties->exitBlock;
    cycleBlockSet.insert(latchBlock);
    cycleBlockSet.insert(loopHeader);

    // Finally, turn it into reduce form.
    SmallVector<Value> iterationValues = transformToReduceLoop(
        loopHeader, exitBlock, cycleBlockSet, getUndefValue, dominanceInfo);

    // Create a block acting as replacement for the loop header and insert
    // the structured loop into it.
    auto *newLoopParentBlock = new Block;
    newLoopParentBlock->insertBefore(loopHeader);
    addBlockArgumentsFromOther(newLoopParentBlock, loopHeader);

    Region::BlockListType &blocks = regionEntry->getParent()->getBlocks();
    Region loopBody;
    // Make sure the loop header is the entry block.
    loopBody.push_back(blocks.remove(loopHeader));
    for (Block *block : cycleBlockSet)
      if (block != latchBlock && block != loopHeader)
        loopBody.push_back(blocks.remove(block));
    // And the latch is the last block.
    loopBody.push_back(blocks.remove(latchBlock));

    Operation *oldTerminator = latchBlock->getTerminator();
    oldTerminator->remove();

    auto builder = OpBuilder::atBlockBegin(newLoopParentBlock);
    FailureOr<Operation *> structuredLoopOp =
        interface.createStructuredDoWhileLoopOp(
            builder, oldTerminator, newLoopParentBlock->getArguments(),
            loopProperties->condition, iterationValues, std::move(loopBody));
    if (failed(structuredLoopOp))
      return failure();
    oldTerminator->erase();

    newSubRegions.push_back(loopHeader);

    for (auto &&[oldValue, newValue] : llvm::zip(
             exitBlock->getArguments(), (*structuredLoopOp)->getResults()))
      oldValue.replaceAllUsesWith(newValue);

    loopHeader->replaceAllUsesWith(newLoopParentBlock);
    // Merge the exit block right after the loop operation.
    newLoopParentBlock->getOperations().splice(newLoopParentBlock->end(),
                                               exitBlock->getOperations());
    exitBlock->erase();
  }
  return newSubRegions;
}

/// Makes sure the branch region only has a single exit. This is required by the
/// recursive part of the algorithm, as it expects the CFG to be single-entry
/// and single-exit. This is done by simply creating an empty block if there
/// is more than one block with an edge to the continuation block. All blocks
/// with edges to the continuation are then redirected to this block. A region
/// terminator is later placed into the block.
static void createSingleExitBranchRegion(
    ArrayRef<Block *> branchRegion, Block *continuation,
    SmallVectorImpl<std::pair<Block *, SmallVector<Value>>> &createdEmptyBlocks,
    Region &conditionalRegion) {
  Block *singleExitBlock = nullptr;
  std::optional<Edge> previousEdgeToContinuation;
  Region::BlockListType &parentBlockList =
      branchRegion.front()->getParent()->getBlocks();
  for (Block *block : branchRegion) {
    for (Edge edge : successorEdges(block)) {
      if (edge.getSuccessor() != continuation)
        continue;

      if (!previousEdgeToContinuation) {
        previousEdgeToContinuation = edge;
        continue;
      }

      // If this is not the first edge to the continuation we create the
      // single exit block and redirect the edges.
      if (!singleExitBlock) {
        singleExitBlock = new Block;
        addBlockArgumentsFromOther(singleExitBlock, continuation);
        previousEdgeToContinuation->setSuccessor(singleExitBlock);
        createdEmptyBlocks.emplace_back(singleExitBlock,
                                        singleExitBlock->getArguments());
      }

      edge.setSuccessor(singleExitBlock);
    }

    conditionalRegion.push_back(parentBlockList.remove(block));
  }

  if (singleExitBlock)
    conditionalRegion.push_back(singleExitBlock);
}

/// Returns true if this block is an exit block of the region.
static bool isRegionExitBlock(Block *block) {
  return block->getNumSuccessors() == 0;
}

/// Transforms the first occurrence of conditional control flow in `regionEntry`
/// into conditionally executed regions. Returns the entry block of the created
/// regions and the region after the conditional control flow.
static FailureOr<SmallVector<Block *>> transformToStructuredCFBranches(
    Block *regionEntry, function_ref<Value(unsigned)> getSwitchValue,
    function_ref<Value(Type)> getUndefValue, CFGToSCFInterface &interface,
    DominanceInfo &dominanceInfo) {
  // Trivial region.
  if (regionEntry->getNumSuccessors() == 0)
    return SmallVector<Block *>{};

  if (regionEntry->getNumSuccessors() == 1) {
    // Single successor we can just splice together.
    Block *successor = regionEntry->getSuccessor(0);
    for (auto &&[oldValue, newValue] : llvm::zip(
             successor->getArguments(), getSuccessorOperands(regionEntry, 0)))
      oldValue.replaceAllUsesWith(newValue);
    regionEntry->getTerminator()->erase();

    regionEntry->getOperations().splice(regionEntry->end(),
                                        successor->getOperations());
    successor->erase();
    return SmallVector<Block *>{regionEntry};
  }

  // Split the CFG into "#numSuccessor + 1" regions.
  // For every edge to a successor, the blocks it solely dominates are
  // determined and become the region following that edge.
  // The last region is the continuation that follows the branch regions.
  SmallPtrSet<Block *, 8> notContinuation;
  notContinuation.insert(regionEntry);
  SmallVector<SmallVector<Block *>> successorBranchRegions(
      regionEntry->getNumSuccessors());
  for (auto &&[blockList, succ] :
       llvm::zip(successorBranchRegions, regionEntry->getSuccessors())) {
    // If the region entry is not the only predecessor, then the edge does not
    // dominate the block it leads to.
    if (succ->getSinglePredecessor() != regionEntry)
      continue;

    // Otherwise get all blocks it dominates in DFS/pre-order.
    DominanceInfoNode *node = dominanceInfo.getNode(succ);
    for (DominanceInfoNode *curr : llvm::depth_first(node)) {
      blockList.push_back(curr->getBlock());
      notContinuation.insert(curr->getBlock());
    }
  }

  // Finds all relevant edges and checks the shape of the control flow graph at
  // this point.
  // Branch regions may either:
  // * Be post-dominated by the continuation
  // * Be post-dominated by a return-like op
  // * Dominate a return-like op and have an edge to the continuation.
  //
  // The control flow graph may then be one of three cases:
  // 1) All branch regions are post-dominated by the continuation. This is the
  // usual case. If there are multiple entry blocks into the continuation a
  // single entry block has to be created. A structured control flow op
  // can then be created from the branch regions.
  //
  // 2) No branch region has an edge to a continuation:
  //                                 +-----+
  //                           +-----+ bb0 +----+
  //                           v     +-----+    v
  //                Region 1 +-+--+    ...     +-+--+ Region n
  //                         |ret1|            |ret2|
  //                         +----+            +----+
  //
  // This can only occur if every region ends with a different kind of
  // return-like op. In that case the control flow operation must stay as we are
  // unable to create a single exit-block. We can nevertheless process all its
  // successors as they single-entry, single-exit regions.
  //
  // 3) Only some branch regions are post-dominated by the continuation.
  // The other branch regions may either be post-dominated by a return-like op
  // or lead to either the continuation or return-like op.
  // In this case we also create a single entry block like in 1) that also
  // includes all edges to the return-like op:
  //                                 +-----+
  //                           +-----+ bb0 +----+
  //                           v     +-----+    v
  //             Region 1    +-+-+    ...     +-+-+ Region n
  //                         +---+            +---+
  //                  +---+  |...              ...
  //                  |ret|<-+ |                |
  //                  +---+    |      +---+     |
  //                           +---->++   ++<---+
  //                                 |     |
  //                                 ++   ++ Region T
  //                                  +---+
  // This transforms to:
  //                                 +-----+
  //                           +-----+ bb0 +----+
  //                           v     +-----+    v
  //                Region 1 +-+-+    ...     +-+-+ Region n
  //                         +---+            +---+
  //                          ...    +-----+   ...
  //                           +---->+ bbM +<---+
  //                                 +-----+
  //                           +-----+  |
  //                           |        v
  //                  +---+    |      +---+
  //                  |ret+<---+     ++   ++
  //                  +---+          |     |
  //                                 ++   ++ Region T
  //                                  +---+
  //
  // bb0 to bbM is now a single-entry, single-exit region that applies to case
  // 1). The control flow op at the end of bbM will trigger case 2.
  SmallVector<Edge> continuationEdges;
  bool continuationPostDominatesAllRegions = true;
  bool noSuccessorHasContinuationEdge = true;
  for (auto &&[entryEdge, branchRegion] :
       llvm::zip(successorEdges(regionEntry), successorBranchRegions)) {

    // If the branch region is empty then the branch target itself is part of
    // the continuation.
    if (branchRegion.empty()) {
      continuationEdges.push_back(entryEdge);
      noSuccessorHasContinuationEdge = false;
      continue;
    }

    for (Block *block : branchRegion) {
      if (isRegionExitBlock(block)) {
        // If a return-like op is part of the branch region then the
        // continuation no longer post-dominates the branch region.
        // Add all its incoming edges to edge list to create the single-exit
        // block for all branch regions.
        continuationPostDominatesAllRegions = false;
        for (auto iter = block->pred_begin(); iter != block->pred_end();
             ++iter) {
          continuationEdges.emplace_back(*iter, iter.getSuccessorIndex());
        }
        continue;
      }

      for (Edge edge : successorEdges(block)) {
        if (notContinuation.contains(edge.getSuccessor()))
          continue;

        continuationEdges.push_back(edge);
        noSuccessorHasContinuationEdge = false;
      }
    }
  }

  // case 2) Keep the control flow op but process its successors further.
  if (noSuccessorHasContinuationEdge)
    return llvm::to_vector(regionEntry->getSuccessors());

  Block *continuation = llvm::find_singleton<Block>(
      continuationEdges, [](Edge edge, bool) { return edge.getSuccessor(); },
      /*AllowRepeats=*/true);

  // In case 3) or if not all continuation edges have the same entry block,
  // create a single entry block as continuation for all branch regions.
  if (!continuation || !continuationPostDominatesAllRegions) {
    EdgeMultiplexer multiplexer = createSingleEntryBlock(
        continuationEdges.front().getFromBlock()->getTerminator()->getLoc(),
        continuationEdges, getSwitchValue, getUndefValue, interface);
    continuation = multiplexer.getMultiplexerBlock();
  }

  // Trigger reprocess of case 3) after creating the single entry block.
  if (!continuationPostDominatesAllRegions) {
    // Unlike in the general case, we are explicitly revisiting the same region
    // entry again after having changed its control flow edges and dominance.
    // We have to therefore explicitly invalidate the dominance tree.
    dominanceInfo.invalidate(regionEntry->getParent());
    return SmallVector<Block *>{regionEntry};
  }

  SmallVector<Block *> newSubRegions;

  // Empty blocks with the values they return to the parent op.
  SmallVector<std::pair<Block *, SmallVector<Value>>> createdEmptyBlocks;

  // Create the branch regions.
  std::vector<Region> conditionalRegions(successorBranchRegions.size());
  for (auto &&[branchRegion, entryEdge, conditionalRegion] :
       llvm::zip(successorBranchRegions, successorEdges(regionEntry),
                 conditionalRegions)) {
    if (branchRegion.empty()) {
      // If no block is part of the branch region, we create a dummy block to
      // place the region terminator into.
      createdEmptyBlocks.emplace_back(
          new Block, llvm::to_vector(entryEdge.getSuccessorOperands()));
      conditionalRegion.push_back(createdEmptyBlocks.back().first);
      continue;
    }

    createSingleExitBranchRegion(branchRegion, continuation, createdEmptyBlocks,
                                 conditionalRegion);

    // The entries of the branch regions may only have redundant block arguments
    // since the edge to the branch region is always dominating.
    Block *subRegionEntryBlock = &conditionalRegion.front();
    for (auto &&[oldValue, newValue] :
         llvm::zip(subRegionEntryBlock->getArguments(),
                   entryEdge.getSuccessorOperands()))
      oldValue.replaceAllUsesWith(newValue);

    subRegionEntryBlock->eraseArguments(0,
                                        subRegionEntryBlock->getNumArguments());
    newSubRegions.push_back(subRegionEntryBlock);
  }

  Operation *structuredCondOp;
  {
    auto opBuilder = OpBuilder::atBlockTerminator(regionEntry);
    FailureOr<Operation *> result = interface.createStructuredBranchRegionOp(
        opBuilder, regionEntry->getTerminator(),
        continuation->getArgumentTypes(), conditionalRegions);
    if (failed(result))
      return failure();
    structuredCondOp = *result;
    regionEntry->getTerminator()->erase();
  }

  for (auto &&[block, valueRange] : createdEmptyBlocks) {
    auto builder = OpBuilder::atBlockEnd(block);
    LogicalResult result = interface.createStructuredBranchRegionTerminatorOp(
        structuredCondOp->getLoc(), builder, structuredCondOp, nullptr,
        valueRange);
    if (failed(result))
      return failure();
  }

  // Any leftover users of the continuation must be from unconditional branches
  // in a branch region. There can only be at most one per branch region as
  // all branch regions have been made single-entry single-exit above.
  // Replace them with the region terminator.
  for (Operation *user : llvm::make_early_inc_range(continuation->getUsers())) {
    assert(user->getNumSuccessors() == 1);
    auto builder = OpBuilder::atBlockTerminator(user->getBlock());
    LogicalResult result = interface.createStructuredBranchRegionTerminatorOp(
        user->getLoc(), builder, structuredCondOp, user,
        static_cast<OperandRange>(
            getMutableSuccessorOperands(user->getBlock(), 0)));
    if (failed(result))
      return failure();
    user->erase();
  }

  for (auto &&[oldValue, newValue] :
       llvm::zip(continuation->getArguments(), structuredCondOp->getResults()))
    oldValue.replaceAllUsesWith(newValue);

  // Splice together the continuations operations with the region entry.
  regionEntry->getOperations().splice(regionEntry->end(),
                                      continuation->getOperations());

  continuation->erase();

  // After splicing the continuation, the region has to be reprocessed as it has
  // new successors.
  newSubRegions.push_back(regionEntry);

  return newSubRegions;
}

/// Transforms the region to only have a single block for every kind of
/// return-like operation that all previous occurrences of the return-like op
/// branch to. If the region only contains a single kind of return-like
/// operation, it creates a single-entry and single-exit region.
static ReturnLikeExitCombiner createSingleExitBlocksForReturnLike(
    Region &region, function_ref<Value(unsigned)> getSwitchValue,
    CFGToSCFInterface &interface) {
  ReturnLikeExitCombiner exitCombiner(region, interface);

  for (Block &block : region.getBlocks()) {
    if (block.getNumSuccessors() != 0)
      continue;
    exitCombiner.combineExit(block.getTerminator(), getSwitchValue);
  }

  return exitCombiner;
}

/// Checks all preconditions of the transformation prior to any transformations.
/// Returns failure if any precondition is violated.
static LogicalResult checkTransformationPreconditions(Region &region) {
  for (Block &block : region.getBlocks())
    if (block.hasNoPredecessors() && !block.isEntryBlock())
      return block.front().emitOpError(
          "transformation does not support unreachable blocks");

  WalkResult result = region.walk([](Operation *operation) {
    if (operation->getNumSuccessors() == 0)
      return WalkResult::advance();

    // This transformation requires all ops with successors to implement the
    // branch op interface. It is impossible to adjust their block arguments
    // otherwise.
    auto branchOpInterface = dyn_cast<BranchOpInterface>(operation);
    if (!branchOpInterface) {
      operation->emitOpError("transformation does not support terminators with "
                             "successors not implementing BranchOpInterface");
      return WalkResult::interrupt();
    }
    // Branch operations must have no side effects. Replacing them would not be
    // valid otherwise.
    if (!isMemoryEffectFree(branchOpInterface)) {
      branchOpInterface->emitOpError(
          "transformation does not support terminators with side effects");
      return WalkResult::interrupt();
    }

    for (unsigned index : llvm::seq(operation->getNumSuccessors())) {
      SuccessorOperands succOps = branchOpInterface.getSuccessorOperands(index);

      // We cannot support operations with operation-produced successor operands
      // as it is currently not possible to pass them to any block arguments
      // other than the first. This breaks creating multiplexer blocks and would
      // likely need special handling elsewhere too.
      if (succOps.getProducedOperandCount() == 0)
        continue;

      branchOpInterface->emitOpError("transformation does not support "
                                     "operations with operation-produced "
                                     "successor operands");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

FailureOr<bool> mlir::transformCFGToSCF(Region &region,
                                        CFGToSCFInterface &interface,
                                        DominanceInfo &dominanceInfo) {
  if (region.empty() || region.hasOneBlock())
    return false;

  if (failed(checkTransformationPreconditions(region)))
    return failure();

  DenseMap<Type, Value> typedUndefCache;
  auto getUndefValue = [&](Type type) {
    auto [iter, inserted] = typedUndefCache.insert({type, nullptr});
    if (!inserted)
      return iter->second;

    auto constantBuilder = OpBuilder::atBlockBegin(&region.front());

    iter->second =
        interface.getUndefValue(region.getLoc(), constantBuilder, type);
    return iter->second;
  };

  // The transformation only creates all values in the range of 0 to
  // max(#numSuccessors). Therefore using a vector instead of a map.
  SmallVector<Value> switchValueCache;
  auto getSwitchValue = [&](unsigned value) {
    if (value < switchValueCache.size())
      if (switchValueCache[value])
        return switchValueCache[value];

    auto constantBuilder = OpBuilder::atBlockBegin(&region.front());

    switchValueCache.resize(
        std::max<size_t>(switchValueCache.size(), value + 1));

    switchValueCache[value] =
        interface.getCFGSwitchValue(region.getLoc(), constantBuilder, value);
    return switchValueCache[value];
  };

  ReturnLikeExitCombiner exitCombiner =
      createSingleExitBlocksForReturnLike(region, getSwitchValue, interface);

  // Invalidate any dominance tree on the region as the exit combiner has
  // added new blocks and edges.
  dominanceInfo.invalidate(&region);

  SmallVector<Block *> workList = {&region.front()};
  while (!workList.empty()) {
    Block *current = workList.pop_back_val();

    // Turn all top-level cycles in the CFG to structured control flow first.
    // After this transformation, the remaining CFG ops form a DAG.
    FailureOr<SmallVector<Block *>> newRegions =
        transformCyclesToSCFLoops(current, getSwitchValue, getUndefValue,
                                  interface, dominanceInfo, exitCombiner);
    if (failed(newRegions))
      return failure();

    // Add the newly created subregions to the worklist. These are the
    // bodies of the loops.
    llvm::append_range(workList, *newRegions);
    // Invalidate the dominance tree as blocks have been moved, created and
    // added during the cycle to structured loop transformation.
    if (!newRegions->empty())
      dominanceInfo.invalidate(current->getParent());

    newRegions = transformToStructuredCFBranches(
        current, getSwitchValue, getUndefValue, interface, dominanceInfo);
    if (failed(newRegions))
      return failure();
    // Invalidating the dominance tree is generally not required by the
    // transformation above as the new region entries correspond to unaffected
    // subtrees in the dominator tree. Only its parent nodes have changed but
    // won't be visited again.
    llvm::append_range(workList, *newRegions);
  }

  return true;
}
