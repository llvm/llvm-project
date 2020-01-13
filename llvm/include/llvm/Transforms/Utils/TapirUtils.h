//===- TapirUtils.h - Utility methods for Tapir ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file utility methods for handling code containing Tapir instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_TAPIRUTILS_H
#define LLVM_TRANSFORMS_UTILS_TAPIRUTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

namespace llvm {

class BasicBlock;
class DominatorTree;
class Loop;
class Task;
class TaskInfo;

/// Returns true if the given instruction performs a detached rethrow, false
/// otherwise.
bool isDetachedRethrow(const Instruction *I, const Value *SyncRegion = nullptr);

/// Returns true if the reattach instruction appears to match the given detach
/// instruction, false otherwise.
bool ReattachMatchesDetach(const ReattachInst *RI, const DetachInst *DI,
                           DominatorTree *DT = nullptr);

// Move static allocas in Block into Entry, which is assumed to dominate
// Block.  Leave lifetime markers behind in Block and before each instruction in
// ExitPoints for those static allocas.  Returns true if Block still contains
// dynamic allocas, which cannot be moved.
bool MoveStaticAllocasInBlock(BasicBlock *Entry, BasicBlock *Block,
                              SmallVectorImpl<Instruction *> &ExitPoints);

/// Serialize the detach DI.  \p ParentEntry should be the entry block of the
/// task that contains DI.  \p Reattaches, \p InlinedLPads, and \p
/// DetachedRethrows identify the reattaches, landing pads, and detached
/// rethrows in the task DI spawns that need special handling during
/// serialization.  If \p DT is provided, then it will be updated to reflect the
/// CFG changes.
void SerializeDetach(DetachInst *DI, BasicBlock *ParentEntry,
                     BasicBlock *EHContinue, Value *LPadValInEHContinue,
                     SmallVectorImpl<Instruction *> &Reattaches,
                     SmallVectorImpl<BasicBlock *> *EHBlocksToClone,
                     SmallPtrSetImpl<BasicBlock *> *EHBlockPreds,
                     SmallPtrSetImpl<LandingPadInst *> *InlinedLPads,
                     SmallVectorImpl<Instruction *> *DetachedRethrows,
                     DominatorTree *DT = nullptr);

/// Analyze a task T for serialization.  Gets the reattaches, landing pads, and
/// detached rethrows that need special handling during serialization.
void AnalyzeTaskForSerialization(
    Task *T, SmallVectorImpl<Instruction *> &Reattaches,
    SmallVectorImpl<BasicBlock *> &EHBlocksToClone,
    SmallPtrSetImpl<BasicBlock *> &EHBlockPreds,
    SmallPtrSetImpl<LandingPadInst *> &InlinedLPads,
    SmallVectorImpl<Instruction *> &DetachedRethrows);

/// Serialize the detach DI that spawns task T.  If \p DT is provided, then it
/// will be updated to reflect the CFG changes.
void SerializeDetach(DetachInst *DI, Task *T, DominatorTree *DT = nullptr);

/// Serialize the sub-CFG detached by the specified detach
/// instruction.  Removes the detach instruction and returns a pointer
/// to the branch instruction that replaces it.
BranchInst *SerializeDetachedCFG(DetachInst *DI, DominatorTree *DT = nullptr);

/// Get the entry basic block to the detached context that contains
/// the specified block.
const BasicBlock *GetDetachedCtx(const BasicBlock *BB);
BasicBlock *GetDetachedCtx(BasicBlock *BB);

/// isCriticalContinueEdge - Return true if the specified edge is a critical
/// detach-continue edge.  Critical detach-continue edges are critical edges -
/// from a block with multiple successors to a block with multiple predecessors
/// - even after ignoring all reattach edges.
bool isCriticalContinueEdge(const Instruction *TI, unsigned SuccNum);

/// GetDetachedCFG - Get the set of basic blocks in the CFG of the parallel task
/// spawned by detach instruction DI.  The CFG will include the
/// exception-handling blocks that are separately identified in EHBlocks, which
/// might not be unique to the task.  TaskReturns will store the set of basic
/// blocks that terminate the CFG of the parallel task.
void GetDetachedCFG(const DetachInst &DI, const DominatorTree &DT,
                    SmallPtrSetImpl<BasicBlock *> &TaskBlocks,
                    SmallPtrSetImpl<BasicBlock *> &EHBlocks,
                    SmallPtrSetImpl<BasicBlock *> &TaskReturns);

/// canDetach - Return true if the given function can perform a detach, false
/// otherwise.
bool canDetach(const Function *F);

/// getDetachUnwindPHIUses - Collect all PHI nodes that directly or indirectly
/// use the landing pad for the unwind destination of detach DI.
void getDetachUnwindPHIUses(DetachInst *DI,
                            SmallPtrSetImpl<BasicBlock *> &UnwindPHIs);

/// Utility class for getting and setting Tapir-related loop hints in the form
/// of loop metadata.
///
/// This class keeps a number of loop annotations locally (as member variables)
/// and can, upon request, write them back as metadata on the loop. It will
/// initially scan the loop for existing metadata, and will update the local
/// values based on information in the loop.
class TapirLoopHints {
public:
  enum SpawningStrategy {
    ST_SEQ,
    ST_DAC,
    ST_END,
  };

private:
  enum HintKind { HK_STRATEGY, HK_GRAINSIZE };

  /// Hint - associates name and validation with the hint value.
  struct Hint {
    const char *Name;
    unsigned Value; // This may have to change for non-numeric values.
    HintKind Kind;

    Hint(const char *Name, unsigned Value, HintKind Kind)
        : Name(Name), Value(Value), Kind(Kind) {}

    bool validate(unsigned Val) const {
      switch (Kind) {
      case HK_STRATEGY:
        return (Val < ST_END);
      case HK_GRAINSIZE:
        return true;
      }
      return false;
    }
  };

  /// Spawning strategy
  Hint Strategy;
  /// Grainsize
  Hint Grainsize;

  /// Return the loop metadata prefix.
  static StringRef Prefix() { return "tapir.loop."; }

public:
  static std::string printStrategy(enum SpawningStrategy Strat) {
    switch(Strat) {
    case TapirLoopHints::ST_SEQ:
      return "Spawn iterations sequentially";
    case TapirLoopHints::ST_DAC:
      return "Use divide-and-conquer";
    case TapirLoopHints::ST_END:
      return "Unknown";
    }
  }

  TapirLoopHints(const Loop *L)
      : Strategy("spawn.strategy", ST_SEQ, HK_STRATEGY),
        Grainsize("grainsize", 0, HK_GRAINSIZE),
        TheLoop(L) {
    // Populate values with existing loop metadata.
    getHintsFromMetadata();
  }

  // /// Dumps all the hint information.
  // std::string emitRemark() const {
  //   TapirLoopReport R;
  //   R << "Strategy = " << printStrategy(getStrategy());

  //   return R.str();
  // }

  enum SpawningStrategy getStrategy() const {
    return (SpawningStrategy)Strategy.Value;
  }

  unsigned getGrainsize() const {
    return Grainsize.Value;
  }

  /// Clear Tapir Hints metadata.
  void clearHintsMetadata();

  /// Mark the loop L as having no spawning strategy.
  void clearStrategy() {
    Strategy.Value = ST_SEQ;
    Hint Hints[] = {Strategy};
    writeHintsToMetadata(Hints);
  }

  void clearClonedLoopMetadata(ValueToValueMapTy &VMap) {
    Hint ClearStrategy = Strategy;
    ClearStrategy.Value = ST_SEQ;
    Hint Hints[] = {ClearStrategy};
    writeHintsToClonedMetadata(Hints, VMap);
  }

  void setAlreadyStripMined() {
    Grainsize.Value = 1;
    Hint Hints[] = {Grainsize};
    writeHintsToMetadata(Hints);
  }

private:
  /// Find hints specified in the loop metadata and update local values.
  void getHintsFromMetadata();

  /// Checks string hint with one operand and set value if valid.
  void setHint(StringRef Name, Metadata *Arg);

  /// Create a new hint from name / value pair.
  MDNode *createHintMetadata(StringRef Name, unsigned V) const;

  /// Matches metadata with hint name.
  bool matchesHintMetadataName(MDNode *Node, ArrayRef<Hint> HintTypes) const;

  /// Sets current hints into loop metadata, keeping other values intact.
  void writeHintsToMetadata(ArrayRef<Hint> HintTypes);

  /// Sets hints into cloned loop metadata, keeping other values intact.
  void writeHintsToClonedMetadata(ArrayRef<Hint> HintTypes,
                                  ValueToValueMapTy &VMap);

  /// The loop these hints belong to.
  const Loop *TheLoop;
};

/// Returns true if Tapir-loop hints require loop outlining during lowering.
bool hintsDemandOutlining(const TapirLoopHints &Hints);

/// Given an loop ID metadata node, returns the loop hint metadata node with the
/// given name (for example, "tapir.loop.stripmine.disable").  If no such
/// metadata node exists, then nullptr is returned.
MDNode *GetStripMineMetadata(MDNode *LoopID, StringRef Name);

/// Examine a given loop to determine if it is a Tapir loop that can and should
/// be processed.  Returns the Task that encodes the loop body if so, or nullptr
/// if not.
Task *getTaskIfTapirLoop(const Loop *L, TaskInfo *TI);

} // End llvm namespace

#endif
