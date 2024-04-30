//===- TileAllocation.cpp - Allocate SME ZA tiles -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform allocates SME tiles at the 'func.func' op level for ArmSME
// operations. It does this using a 16-bit tile mask that has a bit for each
// 128-bit element tile (ZA0.Q-ZA15.Q), the smallest ZA tile granule.
//
// The 128-bit tiles overlap with other element tiles as follows (see section
// B2.3.2 of SME spec [1]):
//
//   Tile    Overlaps
//   ---------------------------------------------------------------------------
//   ZA0.B   ZA0.Q, ZA1.Q, ZA2.Q, ZA3.Q, ZA4.Q, ZA5.Q, ZA6.Q, ZA7.Q, ZA8.Q,
//           ZA9.Q, ZA10.Q, ZA11.Q, ZA12.Q, ZA13.Q, ZA14.Q, ZA15.Q
//   ZA0.H   ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
//   ZA1.H   ZA1.Q, ZA3.Q, ZA5.Q, ZA7.Q, ZA9.Q, ZA11.Q, ZA13.Q, ZA15.Q
//   ZA0.S   ZA0.Q, ZA4.Q, ZA8.Q, ZA12.Q
//   ZA1.S   ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
//   ZA2.S   ZA2.Q, ZA6.Q, ZA10.Q, ZA14.Q
//   ZA3.S   ZA3.Q, ZA7.Q, ZA11.Q, ZA15.Q
//   ZA0.D   ZA0.Q, ZA8.Q
//   ZA1.D   ZA1.Q, ZA9.Q
//   ZA2.D   ZA2.Q, ZA10.Q
//   ZA3.D   ZA3.Q, ZA11.Q
//   ZA4.D   ZA4.Q, ZA12.Q
//   ZA5.D   ZA5.Q, ZA13.Q
//   ZA6.D   ZA6.Q, ZA14.Q
//   ZA7.D   ZA7.Q, ZA15.Q
//
// [1] https://developer.arm.com/documentation/ddi0616/aa
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>

namespace mlir::arm_sme {
#define GEN_PASS_DEF_TESTTILEALLOCATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

enum class TileMask : unsigned {
  // clang-format off
  kZA0B  = 0xffff, // 1111 1111 1111 1111

  kZA0H  = 0xaaaa, // 1010 1010 1010 1010
  kZA1H  = 0x5555, // 0101 0101 0101 0101

  kZA0S  = 0x8888, // 1000 1000 1000 1000
  kZA1S  = 0x4444, // 0100 0100 0100 0100
  kZA2S  = 0x2222, // 0010 0010 0010 0010
  kZA3S  = 0x1111, // 0001 0001 0001 0001

  kZA0D  = 0x8080, // 1000 0000 1000 0000
  kZA1D  = 0x4040, // 0100 0000 0100 0000
  kZA2D  = 0x2020, // 0010 0000 0010 0000
  kZA3D  = 0x1010, // 0001 0000 0001 0000
  kZA4D  = 0x808,  // 0000 1000 0000 1000
  kZA5D  = 0x404,  // 0000 0100 0000 0100
  kZA6D  = 0x202,  // 0000 0010 0000 0010
  kZA7D  = 0x101,  // 0000 0001 0000 0001

  kZA0Q  = 0x8000, // 1000 0000 0000 0000
  kZA1Q  = 0x4000, // 0100 0000 0000 0000
  kZA2Q  = 0x2000, // 0010 0000 0000 0000
  kZA3Q  = 0x1000, // 0001 0000 0000 0000
  kZA4Q  = 0x800,  // 0000 1000 0000 0000
  kZA5Q  = 0x400,  // 0000 0100 0000 0000
  kZA6Q  = 0x200,  // 0000 0010 0000 0000
  kZA7Q  = 0x100,  // 0000 0001 0000 0000
  kZA8Q  = 0x80,   // 0000 0000 1000 0000
  kZA9Q  = 0x40,   // 0000 0000 0100 0000
  kZA10Q = 0x20,   // 0000 0000 0010 0000
  kZA11Q = 0x10,   // 0000 0000 0001 0000
  kZA12Q = 0x8,    // 0000 0000 0000 1000
  kZA13Q = 0x4,    // 0000 0000 0000 0100
  kZA14Q = 0x2,    // 0000 0000 0000 0010
  kZA15Q = 0x1,    // 0000 0000 0000 0001

  kNone = 0x0,     // 0000 0000 0000 0000
  // clang-format on

  LLVM_MARK_AS_BITMASK_ENUM(kZA0B)
};

/// Returns the set of masks relevant for the given type.
static ArrayRef<TileMask> getMasks(ArmSMETileType type) {
  static constexpr std::array ZA_B_MASKS = {TileMask::kZA0B};
  static constexpr std::array ZA_H_MASKS = {TileMask::kZA0H, TileMask::kZA1H};
  static constexpr std::array ZA_S_MASKS = {TileMask::kZA0S, TileMask::kZA1S,
                                            TileMask::kZA2S, TileMask::kZA3S};
  static constexpr std::array ZA_D_MASKS = {
      TileMask::kZA0D, TileMask::kZA1D, TileMask::kZA2D, TileMask::kZA3D,
      TileMask::kZA4D, TileMask::kZA5D, TileMask::kZA6D, TileMask::kZA7D};
  static constexpr std::array ZA_Q_MASKS = {
      TileMask::kZA0Q,  TileMask::kZA1Q,  TileMask::kZA2Q,  TileMask::kZA3Q,
      TileMask::kZA4Q,  TileMask::kZA5Q,  TileMask::kZA6Q,  TileMask::kZA7Q,
      TileMask::kZA8Q,  TileMask::kZA9Q,  TileMask::kZA10Q, TileMask::kZA11Q,
      TileMask::kZA12Q, TileMask::kZA13Q, TileMask::kZA14Q, TileMask::kZA15Q};
  switch (type) {
  case ArmSMETileType::ZAB:
    return ZA_B_MASKS;
  case ArmSMETileType::ZAH:
    return ZA_H_MASKS;
  case ArmSMETileType::ZAS:
    return ZA_S_MASKS;
  case ArmSMETileType::ZAD:
    return ZA_D_MASKS;
  case ArmSMETileType::ZAQ:
    return ZA_Q_MASKS;
  }
}

class TileAllocator {
public:
  /// Allocates and returns a tile ID. Fails if there are no tiles left.
  FailureOr<unsigned> allocateTileId(ArmSMETileType tileType) {
    auto masks = getMasks(tileType);
    for (auto [tileId, tileMask] : llvm::enumerate(masks)) {
      if ((tilesInUse & tileMask) == TileMask::kNone) {
        tilesInUse |= tileMask;
        return tileId;
      }
    }
    return failure();
  }

  /// Releases a previously allocated tile ID.
  void releaseTileId(ArmSMETileType tileType, unsigned tileId) {
    TileMask tileMask = getMasks(tileType)[tileId];
    assert((tilesInUse & tileMask) != TileMask::kNone &&
           "cannot release unallocated tile!");
    tilesInUse ^= tileMask;
  }

  /// Allocates an in-memory tile ID.
  unsigned allocateInMemoryTileId() {
    // Note: We never release in-memory tile IDs. We could, which may allow
    // reusing an allocation, but as we _never_ want to spill an SME tile this
    // is not optimized.
    return nextInMemoryTileId++;
  }

private:
  TileMask tilesInUse = TileMask::kNone;
  unsigned nextInMemoryTileId = kInMemoryTileIdBase;
};

// Add new intermediate blocks for the true and false destinations of a
// `cf.cond_br`. This prevents spurious liveness overlaps due to copies at
// branches.
void splitCondBranches(IRRewriter &rewriter, FunctionOpInterface function) {
  SmallVector<cf::CondBranchOp> worklist;
  function.walk([&](cf::CondBranchOp condBranch) {
    if (llvm::any_of(condBranch->getOperands(), [&](Value value) {
          return isValidSMETileVectorType(value.getType());
        })) {
      worklist.push_back(condBranch);
    }
  });

  auto insertJump = [&](Location loc, Block *source, Block *dest, auto args) {
    rewriter.setInsertionPointToEnd(source);
    rewriter.create<cf::BranchOp>(loc, dest, args);
  };

  for (auto condBranch : worklist) {
    auto loc = condBranch.getLoc();
    Block *block = condBranch->getBlock();
    auto newTrueBranch = rewriter.splitBlock(block, block->end());
    auto newFalseBranch = rewriter.splitBlock(block, block->end());
    insertJump(loc, newTrueBranch, condBranch.getTrueDest(),
               condBranch.getTrueDestOperands());
    insertJump(loc, newFalseBranch, condBranch.getFalseDest(),
               condBranch.getFalseDestOperands());
    condBranch.getFalseDestOperandsMutable().clear();
    condBranch.getTrueDestOperandsMutable().clear();
    condBranch.setSuccessor(newTrueBranch, 0);
    condBranch.setSuccessor(newFalseBranch, 1);
  }
}

/// Inserts tile copies at `cf.br` operations.
void insertCopiesAtBranches(IRRewriter &rewriter,
                            FunctionOpInterface function) {
  splitCondBranches(rewriter, function);
  for (Block &block : function.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (!isa<cf::BranchOp>(terminator))
      continue;
    rewriter.setInsertionPoint(terminator);
    for (OpOperand &operand : terminator->getOpOperands()) {
      if (isValidSMETileVectorType(operand.get().getType())) {
        auto copy =
            rewriter.create<CopyTileOp>(terminator->getLoc(), operand.get());
        operand.assign(copy);
      }
    }
  }
}

/// A range where a tile value is live. The range may contain holes.
struct LiveRange {
  using RangeSet = llvm::IntervalMap<uint64_t, uint8_t, 16,
                                     llvm::IntervalMapHalfOpenInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;
  static constexpr uint8_t kValidLiveRange = 0xff;

  LiveRange(Allocator &allocator)
      : ranges(std::make_unique<RangeSet>(allocator)) {}

  /// Returns true if this range overlaps with `otherRange`.
  bool overlaps(LiveRange const &otherRange) const {
    return llvm::IntervalMapOverlaps<RangeSet, RangeSet>(*ranges,
                                                         *otherRange.ranges)
        .valid();
  }

  /// Unions this live range with `otherRange`, aborts if the ranges overlap.
  void unionWith(LiveRange const &otherRange) {
    for (auto it = otherRange.ranges->begin(); it != otherRange.ranges->end();
         ++it)
      ranges->insert(it.start(), it.stop(), kValidLiveRange);
    values.set_union(otherRange.values);
  }

  /// Inserts an interval [start, end) for `value` into this range.
  void insert(Value value, unsigned start, unsigned end) {
    values.insert(value);
    if (start != end)
      ranges->insert(start, end, kValidLiveRange);
  }

  bool empty() const { return ranges->empty(); }
  unsigned start() const { return ranges->start(); }
  unsigned end() const { return ranges->stop(); }
  bool operator<(LiveRange const &other) const {
    return start() < other.start();
  }

  ArmSMETileType getTileType() const {
    return *getSMETileType(cast<VectorType>(values[0].getType()));
  }

  std::unique_ptr<RangeSet> ranges;
  SetVector<Value> values;
  std::optional<unsigned> tileId;
};

/// Number operations within a function to allow computing live ranges.
DenseMap<Operation *, unsigned>
generateOperationNumbering(FunctionOpInterface function) {
  unsigned index = 0;
  SetVector<Block *> blocks =
      getTopologicallySortedBlocks(function.getFunctionBody());
  DenseMap<Operation *, unsigned> operationToIndexMap;
  for (Block *block : blocks) {
    index++; // We want block args to have their own number.
    for (Operation &op : block->getOperations()) {
      // This is only correct if all ArmSME have been converted to CF.
#ifndef NDEBUG
      op.walk([&](ArmSMETileOpInterface nestedOp) {
        assert(&op == nestedOp.getOperation() &&
               "ArmSME tile allocation does not support nested regions");
      });
#endif
      operationToIndexMap.try_emplace(&op, index++);
    }
  }
  return operationToIndexMap;
}

/// Gather live ranges for SME tiles from the MLIR liveness analysis.
DenseMap<Value, LiveRange>
gatherTileLiveRanges(DenseMap<Operation *, unsigned> const &operationToIndexMap,
                     LiveRange::Allocator &liveRangeAllocator,
                     Liveness &liveness, FunctionOpInterface function) {
  DenseMap<Value, LiveRange> liveRanges;
  auto updateLiveRanges = [&](Value value, Operation *firstUseOrDef,
                              LivenessBlockInfo const &livenessInfo,
                              bool liveAtBlockEntry = false) {
    if (!isValidSMETileVectorType(value.getType()))
      return;
    auto it = liveRanges.try_emplace(value, liveRangeAllocator).first;
    auto lastUseInBlock = livenessInfo.getEndOperation(value, firstUseOrDef);
    unsigned start =
        operationToIndexMap.at(firstUseOrDef) + (liveAtBlockEntry ? -1 : 0);
    unsigned end = operationToIndexMap.at(lastUseInBlock);
    it->second.insert(value, start, end);
  };

  for (Block &block : function.getBlocks()) {
    LivenessBlockInfo const *livenessInfo = liveness.getLiveness(&block);
    // Handle block arguments:
    for (Value argument : block.getArguments())
      updateLiveRanges(argument, &block.front(), *livenessInfo,
                       /*liveAtBlockEntry=*/true);
    // Handle live-ins:
    for (Value liveIn : livenessInfo->in())
      updateLiveRanges(liveIn, &block.front(), *livenessInfo,
                       /*liveAtBlockEntry=*/true);
    // Handle new definitions:
    for (Operation &op : block) {
      for (Value result : op.getResults())
        updateLiveRanges(result, &op, *livenessInfo);
    }
  }

  return liveRanges;
}

/// Iterate over all predecessor tile values to a (tile) block argument.
static void forEachPredecessorTileValue(BlockArgument blockArg,
                                        function_ref<void(Value)> callback) {
  Block *block = blockArg.getOwner();
  unsigned argNumber = blockArg.getArgNumber();
  for (Block *pred : block->getPredecessors()) {
    TypeSwitch<Operation *>(pred->getTerminator())
        .Case<cf::BranchOp>([&](auto branch) {
          Value predecessorOperand = branch.getDestOperands()[argNumber];
          callback(predecessorOperand);
        })
        .Case<cf::CondBranchOp>([&](auto condBranch) {
          if (condBranch.getFalseDest() == block) {
            Value predecessorOperand =
                condBranch.getFalseDestOperands()[argNumber];
            callback(predecessorOperand);
          }
          if (condBranch.getTrueDest() == block) {
            Value predecessorOperand =
                condBranch.getTrueDestOperands()[argNumber];
            callback(predecessorOperand);
          }
        });
  }
}

/// Coalesce live ranges where it would prevent unnecessary tile moves.
SmallVector<LiveRange *>
coalesceTileLiveRanges(DenseMap<Value, LiveRange> &initialLiveRanges) {
  DenseMap<Value, LiveRange *> liveRanges;
  for (auto &[value, liveRange] : initialLiveRanges) {
    liveRanges.insert({value, &liveRange});
  }

  auto mergeValuesIfNonOverlapping = [&](Value a, Value b) {
    LiveRange *aLiveRange = liveRanges.at(a);
    LiveRange *bLiveRange = liveRanges.at(b);
    if (aLiveRange != bLiveRange && !aLiveRange->overlaps(*bLiveRange)) {
      aLiveRange->unionWith(*bLiveRange);
      for (Value value : bLiveRange->values)
        liveRanges[value] = aLiveRange;
    }
  };

  // Merge the live ranges of new definitions with their tile operands.
  auto unifyDefinitionsWithOperands = [&](Value value) {
    auto armSMEOp = value.getDefiningOp<ArmSMETileOpInterface>();
    if (!armSMEOp)
      return;
    for (auto operand : armSMEOp->getOperands()) {
      if (isValidSMETileVectorType(operand.getType()))
        mergeValuesIfNonOverlapping(value, operand);
    }
  };

  // Merge the live ranges of block arguments with their predecessors.
  auto unifyBlockArgumentsWithPredecessors = [&](Value value) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return;
    forEachPredecessorTileValue(blockArg, [&](Value predecessorTile) {
      mergeValuesIfNonOverlapping(blockArg, predecessorTile);
    });
  };

  auto applyRule = [&](auto rule) {
    llvm::for_each(llvm::make_first_range(initialLiveRanges), rule);
  };

  // Unify as many live ranges as we can. This prevents unnecessary moves.
  applyRule(unifyBlockArgumentsWithPredecessors);
  applyRule(unifyDefinitionsWithOperands);

  // Remove duplicate live range entries.
  SetVector<LiveRange *> uniqueLiveRanges;
  for (auto [_, liveRange] : liveRanges) {
    if (!liveRange->empty())
      uniqueLiveRanges.insert(liveRange);
  }

  // Sort the new live ranges by starting point (ready for tile allocation).
  auto coalescedLiveRanges = uniqueLiveRanges.takeVector();
  std::sort(coalescedLiveRanges.begin(), coalescedLiveRanges.end(),
            [](LiveRange *a, LiveRange *b) { return *a < *b; });
  return std::move(coalescedLiveRanges);
}

/// Greedily allocate tile IDs to live ranges spill using simple heuristics.
/// Note: This does not attempt to fill holes in live/allocated ranges.
void allocateTilesToLiveRanges(ArrayRef<LiveRange *> liveRanges) {
  TileAllocator tileAllocator;
  SetVector<LiveRange *> allocatedRanges;

  auto chooseSpillUsingHeuristics = [&](LiveRange *newRange) {
    unsigned memoryTileId = tileAllocator.allocateInMemoryTileId();
    auto spillActiveRange = [&](LiveRange *range) {
      unsigned tileId = *range->tileId;
      range->tileId = memoryTileId;
      allocatedRanges.remove(range);
      return tileId;
    };

    auto isTrivialSpill = [](LiveRange *allocatedRange) {
      return allocatedRange->values.size() == 1 &&
             isTriviallyCloneableTileOp(
                 allocatedRange->values[0]
                     .getDefiningOp<ArmSMETileOpInterface>());
    };

    // Heuristic: Spill trivially copyable operations (usually free).
    if (isTrivialSpill(newRange))
      return memoryTileId;
    auto trivialSpill = llvm::find_if(allocatedRanges, isTrivialSpill);
    if (trivialSpill != allocatedRanges.end())
      return spillActiveRange(*trivialSpill);

    // Heuristic: Spill the live range that ends last.
    LiveRange *lastActiveLiveRange = *std::max_element(
        allocatedRanges.begin(), allocatedRanges.end(),
        [](LiveRange *a, LiveRange *b) { return a->end() < b->end(); });
    if (lastActiveLiveRange->end() >= newRange->end())
      return spillActiveRange(lastActiveLiveRange);

    return memoryTileId;
  };

  for (LiveRange *newRange : liveRanges) {
    // Release tiles from live ranges that have ended.
    allocatedRanges.remove_if([&](LiveRange *allocatedRange) {
      if (allocatedRange->end() <= newRange->start()) {
        tileAllocator.releaseTileId(allocatedRange->getTileType(),
                                    *allocatedRange->tileId);
        return true;
      }
      return false;
    });

    // Allocate a tile ID to `newRange`.
    auto tileId = tileAllocator.allocateTileId(newRange->getTileType());
    if (succeeded(tileId))
      newRange->tileId = *tileId;
    else
      newRange->tileId = chooseSpillUsingHeuristics(newRange);

    // Insert the live range into the allocated ranges.
    if (newRange->tileId < kInMemoryTileIdBase)
      allocatedRanges.insert(newRange);
  }
}

/// Assign tile IDs back to IR and attempt to resolve trivial tile ID conflicts.
LogicalResult assignTileIdsAndResolveTrivialConflicts(
    IRRewriter &rewriter, FunctionOpInterface function,
    ArrayRef<LiveRange *> allocatedLiveRanges) {
  for (LiveRange const *liveRange : allocatedLiveRanges) {
    auto tileIdAttr = rewriter.getI32IntegerAttr(*liveRange->tileId);
    auto isAllocatedToSameTile = [&](Value value) {
      if (auto tileOp = value.getDefiningOp<ArmSMETileOpInterface>();
          tileOp && tileOp.getTileId() == tileIdAttr)
        return true;
      return liveRange->values.contains(value);
    };
    for (Value value : liveRange->values) {
      for (Operation *user : value.getUsers()) {
        if (auto tileOp = dyn_cast<ArmSMETileOpInterface>(user)) {
          // Ensure ArmSME ops that don't produce a value still get a tile ID.
          if (!hasTileResult(tileOp))
            tileOp.setTileId(tileIdAttr);
        }
      }
      auto copyOp = value.getDefiningOp<CopyTileOp>();
      if (copyOp && isAllocatedToSameTile(copyOp.getTile())) {
        // Fold redundant copies.
        rewriter.replaceAllUsesWith(copyOp, copyOp.getTile());
      } else if (auto tileOp = value.getDefiningOp<ArmSMETileOpInterface>()) {
        tileOp.setTileId(tileIdAttr);
        // Rectify operand tile IDs with result tile IDs.
        OpOperand *tileOperand = getTileOpOperand(tileOp);
        if (!tileOperand || isAllocatedToSameTile(tileOperand->get()))
          continue;
        auto operandTileOp =
            tileOperand->get().getDefiningOp<ArmSMETileOpInterface>();
        if (!isTriviallyCloneableTileOp(operandTileOp))
          return tileOp.emitOpError("failed to rectify tile operand with tile "
                                    "result (move required)");
        // Cloning prevents a move/spill (though may require recomputation).
        rewriter.setInsertionPoint(tileOp);
        auto clonedOp = operandTileOp.clone();
        clonedOp.setTileId(tileOp.getTileId());
        rewriter.insert(clonedOp);
        if (copyOp)
          rewriter.replaceAllUsesWith(copyOp, clonedOp->getResult(0));
        else
          tileOperand->assign(clonedOp->getResult(0));
      } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        // Validate block arguments.
        bool tileMismatch = false;
        forEachPredecessorTileValue(blockArg, [&](Value predecessorTile) {
          if (tileMismatch)
            return;
          if (!isAllocatedToSameTile(predecessorTile)) {
            blockArg.getOwner()->getParentOp()->emitOpError(
                "block argument not allocated to the same tile as "
                "predecessors");
            tileMismatch = true;
          }
        });
        if (tileMismatch)
          return failure();
      }
    }
  }
  return success();
}

/// Prints live ranges alongside operation names for debugging.
void dumpLiveRanges(DenseMap<Operation *, unsigned> const &operationToIndexMap,
                    ArrayRef<LiveRange const *> liveRanges,
                    FunctionOpInterface function) {
  llvm::errs() << "SME Tile Liveness: @" << function.getName()
               << "\nKey:\nS - Start\nE - End\n| - Live\n";
  for (auto [blockIdx, block] : llvm::enumerate(function.getBlocks())) {
    llvm::errs() << "^bb" << blockIdx << ":\n";
    for (Operation &op : block.getOperations()) {
      unsigned operationIndex = operationToIndexMap.at(&op);
      for (LiveRange const *range : liveRanges) {
        char liveness = ' ';
        for (auto it = range->ranges->begin(); it != range->ranges->end();
             ++it) {
          if (it.start() == operationIndex)
            liveness = (liveness == 'E' ? '|' : 'S');
          else if (it.stop() == operationIndex)
            liveness = (liveness == 'S' ? '|' : 'E');
          else if (operationIndex >= it.start() && operationIndex < it.stop())
            liveness = '|';
        }
        llvm::errs() << liveness;
      }
      llvm::errs() << ' ' << op.getName() << '\n';
    }
  }
  llvm::errs() << "==========\n";
}

struct TestTileAllocationPass
    : public arm_sme::impl::TestTileAllocationBase<TestTileAllocationPass> {
  using TestTileAllocationBase::TestTileAllocationBase;
  void runOnOperation() override {
    if (failed(arm_sme::allocateSMETiles(getOperation(), dumpTileLiveRanges)))
      signalPassFailure();
  }
};
} // namespace

LogicalResult mlir::arm_sme::allocateSMETiles(FunctionOpInterface function,
                                              bool dumpRanges) {
  LiveRange::Allocator liveRangeAllocator;
  IRRewriter rewriter(function.getContext());

  // 1. Insert copy operations at branch operations.
  insertCopiesAtBranches(rewriter, function);

  // 2. Gather live ranges for each ArmSME tile within the function.
  Liveness liveness(function);
  auto operationToIndexMap = generateOperationNumbering(function);
  auto initialLiveRanges = gatherTileLiveRanges(
      operationToIndexMap, liveRangeAllocator, liveness, function);
  if (initialLiveRanges.empty())
    return success();

  if (dumpRanges) {
    // Wrangle initial live ranges into a form suitable for printing.
    auto nonEmpty = llvm::make_filter_range(
        llvm::make_second_range(initialLiveRanges),
        [&](LiveRange const &liveRange) { return !liveRange.empty(); });
    auto initialRanges = llvm::to_vector(llvm::map_range(
        nonEmpty, [](LiveRange const &liveRange) { return &liveRange; }));
    std::sort(initialRanges.begin(), initialRanges.end(),
              [](LiveRange const *a, LiveRange const *b) { return *a < *b; });
    llvm::errs() << "\n========== Initial Live Ranges:\n";
    dumpLiveRanges(operationToIndexMap, initialRanges, function);
  }

  // 3. Coalesce (non-overlapping) live ranges where it would be beneficial
  // for tile allocation. E.g. Unify the result of an operation with its
  // operands.
  auto coalescedLiveRanges = coalesceTileLiveRanges(initialLiveRanges);

  if (dumpRanges) {
    llvm::errs() << "\n========== Coalesced Live Ranges:\n";
    dumpLiveRanges(operationToIndexMap, coalescedLiveRanges, function);
  }

  // 4. Allocate tile IDs to live ranges.
  allocateTilesToLiveRanges(coalescedLiveRanges);

  // 5. Assign the tile IDs back to the ArmSME operations.
  if (failed(assignTileIdsAndResolveTrivialConflicts(rewriter, function,
                                                     coalescedLiveRanges))) {
    return failure();
  }

  /// 6. Erase trivially dead tile operations (e.g. a ZeroOp with no
  /// users). This prevents the LLVM conversion needlessly inserting spills.
  eraseTriviallyDeadTileOps(rewriter, function);
  return success();
}
