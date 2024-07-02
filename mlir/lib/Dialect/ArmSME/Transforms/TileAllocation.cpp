//===- TileAllocation.cpp - Allocate SME ZA tiles -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform allocates SME tiles at the 'func.func' op level for ArmSME
// operations. It roughly implements a linear scan register allocator, similar
// to the one outlined in [1], but with simplifications and assumptions made for
// our use case. Note that this is a greedy allocator (so it may not always find
// the most optimal allocation of tiles).
//
// The allocator operates at the CF dialect level. It is the responsibility of
// users to ensure the IR has been lowered to CF before invoking the tile
// allocator.
//
// The 128-bit tiles overlap with other element tiles as follows (see section
// B2.3.2 of SME spec [2]):
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
// [1] "Linear Scan Register Allocation in the Context of SSA Form and Register
//      Constraints" (Hanspeter Mössenböck and Michael Pfeiffer)
//     https://link.springer.com/content/pdf/10.1007/3-540-45937-5_17.pdf
// [2] https://developer.arm.com/documentation/ddi0616/aa
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
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

/// Add new intermediate blocks for the true and false destinations of
/// `cf.cond_br`s that contain tile operands. This prevents spurious liveness
/// overlaps due to copies at branches.
///
///  BEFORE:
///  ```mlir
///  cf.cond_br %cond, ^bb1(%tile: vector<[4]x[4]xf32>), ^bb2
///  ```
///
///  AFTER:
///  ```mlir
///    cf.cond_br %cond, ^bb1_copy, ^bb2_copy
///  ^bb1_copy:
///    cf.br ^bb1(%tile: vector<[4]x[4]xf32>)
///  ^bb2_copy:
///    cf.br ^bb2
///  ```
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
    rewriter.modifyOpInPlace(condBranch, [&] {
      condBranch.getFalseDestOperandsMutable().clear();
      condBranch.getTrueDestOperandsMutable().clear();
      condBranch.setSuccessor(newTrueBranch, 0);
      condBranch.setSuccessor(newFalseBranch, 1);
    });
  }
}

/// Inserts tile copies at `cf.br` operations.
///
///  BEFORE:
///  ```mlir
///  cf.br ^bb1(%tile: vector<[4]x[4]xf32>)
///  ```
///
///  AFTER:
///  ```mlir
///  %copy = arm_sme.copy_tile %tile : vector<[4]x[4]xf32>
///  cf.br ^bb1(%copy: vector<[4]x[4]xf32>)
///  ```
void insertCopiesAtBranches(IRRewriter &rewriter,
                            FunctionOpInterface function) {
  for (Block &block : function.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (!isa<cf::BranchOp>(terminator))
      continue;
    rewriter.setInsertionPoint(terminator);
    for (OpOperand &operand : terminator->getOpOperands()) {
      if (isValidSMETileVectorType(operand.get().getType())) {
        auto copy =
            rewriter.create<CopyTileOp>(terminator->getLoc(), operand.get());
        rewriter.modifyOpInPlace(terminator, [&] { operand.assign(copy); });
      }
    }
  }
}

/// Prepares the IR for tile allocation. It does this by first 'splitting'
/// conditional branches (see `splitCondBranches`), then inserting tile copies
/// at branch operations. The conditional branches are split to prevent the
/// copies needed for them overlapping between the true and false paths of the
/// branch (see `tile-allocation-copies.mlir` and
/// `tile-allocation-liveness.mlir` for examples). The copies break up live
/// ranges and ensure when moving out of SSA the semantics of the program are
/// preserved.
void preprocessForTileAllocation(IRRewriter &rewriter,
                                 FunctionOpInterface function) {
  splitCondBranches(rewriter, function);
  insertCopiesAtBranches(rewriter, function);
}

/// A live range for a (collection of) tile values. A live range is built up of
/// non-overlapping intervals [start, end) which represent parts of the program
/// where a value in the range needs to be live (i.e. in an SME virtual tile).
/// Note that as the intervals are non-overlapping all values within a live
/// range can be allocated to the same SME virtual tile.
struct LiveRange {
  using RangeSet = llvm::IntervalMap<uint64_t, uint8_t, 16,
                                     llvm::IntervalMapHalfOpenInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;
  // Dummy value for the IntervalMap. Only the keys matter (the intervals).
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

  /// The values contained in this live range.
  SetVector<Value> values;

  /// A set of (non-overlapping) intervals that mark where any value in `values`
  /// is live.
  std::unique_ptr<RangeSet> ranges;

  /// The tile ID (or none) assigned to this live range.
  std::optional<unsigned> tileId;
};

/// Number operations within a function to allow computing live ranges.
/// Operations are numbered consecutively wihin blocks, and the blocks are
/// topologically sorted (using forward edges). This function is only correct if
/// all ArmSME have been converted to CF (which is asserted).
DenseMap<Operation *, unsigned>
generateOperationNumbering(FunctionOpInterface function) {
  unsigned index = 0;
  SetVector<Block *> blocks =
      getBlocksSortedByDominance(function.getFunctionBody());
  DenseMap<Operation *, unsigned> operationToIndexMap;
  for (Block *block : blocks) {
    index++; // We want block args to have their own number.
    for (Operation &op : block->getOperations()) {
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
  assert(!operationToIndexMap.empty() && "expected operation numbering");
  DenseMap<Value, LiveRange> liveRanges;
  /// Defines or updates a live range for an SME tile value. Live-ins may update
  /// an existing live range (rather than define a new one). Note: If
  /// `liveAtBlockEntry` is true then `firstUseOrDef` is the first operation in
  /// the block.
  auto defineOrUpdateValueLiveRange = [&](Value value, Operation *firstUseOrDef,
                                          LivenessBlockInfo const &livenessInfo,
                                          bool liveAtBlockEntry = false) {
    if (!isValidSMETileVectorType(value.getType()))
      return;
    // Find or create a live range for `value`.
    auto [it, _] = liveRanges.try_emplace(value, liveRangeAllocator);
    LiveRange &valueLiveRange = it->second;
    auto lastUseInBlock = livenessInfo.getEndOperation(value, firstUseOrDef);
    // Add the interval [firstUseOrDef, lastUseInBlock) to the live range.
    unsigned startOpIdx =
        operationToIndexMap.at(firstUseOrDef) + (liveAtBlockEntry ? -1 : 0);
    unsigned endOpIdx = operationToIndexMap.at(lastUseInBlock);
    valueLiveRange.insert(value, startOpIdx, endOpIdx);
  };

  for (Block &block : function.getBlocks()) {
    LivenessBlockInfo const *livenessInfo = liveness.getLiveness(&block);
    // Handle block arguments:
    for (Value argument : block.getArguments())
      defineOrUpdateValueLiveRange(argument, &block.front(), *livenessInfo,
                                   /*liveAtBlockEntry=*/true);
    // Handle live-ins:
    for (Value liveIn : livenessInfo->in())
      defineOrUpdateValueLiveRange(liveIn, &block.front(), *livenessInfo,
                                   /*liveAtBlockEntry=*/true);
    // Handle new definitions:
    for (Operation &op : block) {
      for (Value result : op.getResults())
        defineOrUpdateValueLiveRange(result, &op, *livenessInfo);
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

  // Merge the live ranges of values `a` and `b` into one (if they do not
  // overlap). After this, the values `a` and `b` will both point to the same
  // live range (which will contain multiple values).
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

/// Choose a live range to spill (via some heuristics). This picks either an
/// active live range from `activeRanges` or the new live range `newRange`.
LiveRange *chooseSpillUsingHeuristics(ArrayRef<LiveRange *> activeRanges,
                                      LiveRange *newRange) {
  // Heuristic: Spill trivially copyable operations (usually free).
  auto isTrivialSpill = [&](LiveRange *allocatedRange) {
    return isTileTypeGreaterOrEqual(allocatedRange->getTileType(),
                                    newRange->getTileType()) &&
           allocatedRange->values.size() == 1 &&
           isTriviallyCloneableTileOp(
               allocatedRange->values[0]
                   .getDefiningOp<ArmSMETileOpInterface>());
  };
  if (isTrivialSpill(newRange))
    return newRange;
  auto trivialSpill = llvm::find_if(activeRanges, isTrivialSpill);
  if (trivialSpill != activeRanges.end())
    return *trivialSpill;

  // Heuristic: Spill the range that ends last (with a compatible tile type).
  auto isSmallerTileTypeOrEndsEarlier = [](LiveRange *a, LiveRange *b) {
    return !isTileTypeGreaterOrEqual(a->getTileType(), b->getTileType()) ||
           a->end() < b->end();
  };
  LiveRange *lastActiveLiveRange = *std::max_element(
      activeRanges.begin(), activeRanges.end(), isSmallerTileTypeOrEndsEarlier);
  if (!isSmallerTileTypeOrEndsEarlier(lastActiveLiveRange, newRange))
    return lastActiveLiveRange;
  return newRange;
}

/// Greedily allocate tile IDs to live ranges. Spill using simple heuristics.
/// Note: This does not attempt to fill holes in active live ranges.
void allocateTilesToLiveRanges(
    ArrayRef<LiveRange *> liveRangesSortedByStartPoint) {
  TileAllocator tileAllocator;
  SetVector<LiveRange *> activeRanges;
  for (LiveRange *nextRange : liveRangesSortedByStartPoint) {
    // Release tile IDs from live ranges that have ended.
    activeRanges.remove_if([&](LiveRange *activeRange) {
      if (activeRange->end() <= nextRange->start()) {
        tileAllocator.releaseTileId(activeRange->getTileType(),
                                    *activeRange->tileId);
        return true;
      }
      return false;
    });

    // Allocate a tile ID to `nextRange`.
    auto rangeTileType = nextRange->getTileType();
    auto tileId = tileAllocator.allocateTileId(rangeTileType);
    if (succeeded(tileId)) {
      nextRange->tileId = *tileId;
    } else {
      LiveRange *rangeToSpill =
          chooseSpillUsingHeuristics(activeRanges.getArrayRef(), nextRange);
      if (rangeToSpill != nextRange) {
        // Spill an active live range (so release its tile ID first).
        tileAllocator.releaseTileId(rangeToSpill->getTileType(),
                                    *rangeToSpill->tileId);
        activeRanges.remove(rangeToSpill);
        // This will always succeed after a spill (of an active live range).
        nextRange->tileId = *tileAllocator.allocateTileId(rangeTileType);
      }
      rangeToSpill->tileId = tileAllocator.allocateInMemoryTileId();
    }

    // Insert the live range into the active ranges.
    if (nextRange->tileId < kInMemoryTileIdBase)
      activeRanges.insert(nextRange);
  }
}

/// Assigns a tile ID to an MLIR value.
void assignTileIdToValue(IRRewriter &rewriter, Value value,
                         IntegerAttr tileIdAttr) {
  if (auto tileOp = value.getDefiningOp<ArmSMETileOpInterface>())
    rewriter.modifyOpInPlace(tileOp, [&] { tileOp.setTileId(tileIdAttr); });
  for (Operation *user : value.getUsers()) {
    if (auto tileOp = dyn_cast<ArmSMETileOpInterface>(user)) {
      // Ensure ArmSME ops that don't produce a value still get a tile ID.
      if (!hasTileResult(tileOp))
        rewriter.modifyOpInPlace(tileOp, [&] { tileOp.setTileId(tileIdAttr); });
    }
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

    /// Eliminates copies where the operand has the same tile ID.
    auto foldRedundantCopies = [&](Value value) -> LogicalResult {
      auto copyOp = value.getDefiningOp<CopyTileOp>();
      if (!copyOp || !isAllocatedToSameTile(copyOp.getTile()))
        return failure();
      rewriter.replaceAllUsesWith(copyOp, copyOp.getTile());
      return success();
    };

    /// Validates each predecessor to a tile block argument has been assigned
    /// the same tile ID.
    auto validateBlockArguments = [&](Value value) {
      auto blockArg = dyn_cast<BlockArgument>(value);
      if (!blockArg) {
        // Not a block argument (nothing to validate).
        return success();
      }
      bool tileMismatch = false;
      forEachPredecessorTileValue(blockArg, [&](Value predecessorTile) {
        if (tileMismatch)
          return;
        if (!isAllocatedToSameTile(predecessorTile)) {
          blockArg.getOwner()->getParentOp()->emitOpError(
              "block argument not allocated to the same SME virtial tile as "
              "predecessors");
          tileMismatch = true;
        }
      });
      return success(/*isSuccess=*/!tileMismatch);
    };

    /// Attempts to resolve (trivial) tile ID conflicts.
    auto resolveTrivialTileConflicts = [&](Value value) -> LogicalResult {
      auto tileOp = value.getDefiningOp<ArmSMETileOpInterface>();
      OpOperand *tileOperand = getTileOpOperand(tileOp);
      if (!tileOperand || isAllocatedToSameTile(tileOperand->get())) {
        // Operand already allocated to the correct tile.
        // No conflict to resolve.
        return success();
      }
      auto operandTileOp =
          tileOperand->get().getDefiningOp<ArmSMETileOpInterface>();
      if (!isTriviallyCloneableTileOp(operandTileOp)) {
        auto error =
            tileOp.emitOpError("tile operand allocated to different SME "
                               "virtial tile (move required)");
        error.attachNote(tileOperand->get().getLoc())
            << "tile operand is: " << tileOperand->get();
        return error;
      }
      // Cloning prevents a move/spill (though may require recomputation).
      rewriter.setInsertionPoint(tileOp);
      auto clonedOp = operandTileOp.clone();
      rewriter.modifyOpInPlace(clonedOp,
                               [&] { clonedOp.setTileId(tileOp.getTileId()); });
      rewriter.insert(clonedOp);
      if (isa<CopyTileOp>(tileOp)) {
        rewriter.replaceAllUsesWith(tileOp->getResult(0),
                                    clonedOp->getResult(0));
      } else {
        rewriter.modifyOpInPlace(
            tileOp, [&] { tileOperand->assign(clonedOp->getResult(0)); });
      }
      return success();
    };

    for (Value value : liveRange->values) {
      // 1. Assign the tile ID to the value.
      assignTileIdToValue(rewriter, value, tileIdAttr);

      // 2. Attempt to eliminate redundant tile copies.
      if (succeeded(foldRedundantCopies(value)))
        continue;

      // 3. Validate tile block arguments.
      if (failed(validateBlockArguments(value)))
        return failure();

      // 4. Attempt to resolve (trivial) tile ID conflicts.
      if (failed(resolveTrivialTileConflicts(value)))
        return failure();
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
    FunctionOpInterface function = getOperation();
    if (preprocessOnly) {
      IRRewriter rewriter(function);
      return preprocessForTileAllocation(rewriter, function);
    }
    if (failed(arm_sme::allocateSMETiles(function, dumpTileLiveRanges)))
      signalPassFailure();
  }
};
} // namespace

LogicalResult mlir::arm_sme::allocateSMETiles(FunctionOpInterface function,
                                              bool dumpRanges) {
  if (function.empty()) {
    // TODO: Also return early if the function contains no ArmSME ops?
    return success();
  }

  LiveRange::Allocator liveRangeAllocator;
  IRRewriter rewriter(function.getContext());

  // 1. Preprocess the IR for tile allocation.
  preprocessForTileAllocation(rewriter, function);

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

  // 6. Erase trivially dead tile operations (e.g. a ZeroOp with no
  // users). This prevents the LLVM conversion needlessly inserting spills.
  eraseTriviallyDeadTileOps(rewriter, function);
  return success();
}
