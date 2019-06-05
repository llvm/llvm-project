//===---- LoopSpawning.h - Spawn loop iterations efficiently ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass modifies Tapir loops to spawn their iterations efficiently.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
#define LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"

#define LS_NAME "loop-spawning"

namespace llvm {

class AssumptionCache;
class DominatorTree;
class Function;
class LoopAccessInfo;
class LoopInfo;
class OptimizationRemarkEmitter;
class PHINode;
class Value;
class ScalarEvolution;
class SyncInst;
class TargetTransformInfo;
class Type;

/// LoopOutline serves as a base class for different variants of LoopSpawning.
/// LoopOutline implements common parts of LoopSpawning transformations, namely,
/// lifting a Tapir loop into a separate helper function.
class LoopOutline {
public:
  LoopOutline(Loop *OrigLoop, ScalarEvolution &SE,
              LoopInfo *LI, DominatorTree *DT,
              AssumptionCache *AC,
              OptimizationRemarkEmitter &ORE)
      : OrigLoop(OrigLoop), SE(SE), LI(LI), DT(DT), AC(AC), ORE(ORE),
        ExitBlock(nullptr)
  {
    // Use the loop latch to determine the canonical exit block for this loop.
    Instruction *TI = OrigLoop->getLoopLatch()->getTerminator();
    if (2 != TI->getNumSuccessors())
      return;
    ExitBlock = TI->getSuccessor(0);
    if (ExitBlock == OrigLoop->getHeader())
      ExitBlock = TI->getSuccessor(1);
  }

  virtual bool processLoop() = 0;

  virtual ~LoopOutline() {}

protected:
  /// \brief Helper routine to get all exit blocks of a loop.
  static void getEHExits(Loop *L, const BasicBlock *DesignatedExitBlock,
                         const BasicBlock *DetachUnwind,
                         const Value *SyncRegion,
                         SmallVectorImpl<BasicBlock *> &EHExits);

  /// \brief Get the wider of two integer types.
  static Type *getWiderType(const DataLayout &DL, Type *Ty0, Type *Ty1);

  /// \brief Canonicalize the induction variables in the loop.  Return the
  /// canonical induction variable created or inserted by the scalar evolution
  /// expander.
  PHINode *canonicalizeIVs(Type *Ty);

  /// \brief Replace the latch of the loop to check that IV is always less than
  /// or equal to the limit.
  Value *canonicalizeLoopLatch(PHINode *IV, Value *Limit);

  /// \brief Returns true if the specified value is used anywhere in the given
  /// set LoopBlocks other than Latch.  Returns false otherwise.
  bool isUsedInLoopBody(const Value *V, std::vector<BasicBlock *> &LoopBlocks,
                        const Instruction *Cond);

  /// \brief Insert a sync before the specified escape, which is either a return
  /// or a resume.
  static SyncInst *insertSyncBeforeEscape(BasicBlock *Esc, Instruction *SyncReg,
                                          DominatorTree *DT, LoopInfo *LI);

  /// \brief Unlink the specified loop, and update analysis accordingly.
  void unlinkLoop();

  /// The original loop.
  Loop *OrigLoop;

  /// A wrapper around ScalarEvolution used to add runtime SCEV checks. Applies
  /// dynamic knowledge to simplify SCEV expressions and converts them to a
  /// more usable form.
  // PredicatedScalarEvolution &PSE;
  ScalarEvolution &SE;
  /// Loop info.
  LoopInfo *LI;
  /// Dominator tree.
  DominatorTree *DT;
  /// Assumption cache.
  AssumptionCache *AC;
  /// Interface to emit optimization remarks.
  OptimizationRemarkEmitter &ORE;

  /// The exit block of this loop.  We compute our own exit block, based on the
  /// latch, and handle other exit blocks (i.e., for exception handling) in a
  /// special manner.
  BasicBlock *ExitBlock;

// private:
//   /// Report an analysis message to assist the user in diagnosing loops that are
//   /// not transformed.  These are handled as LoopAccessReport rather than
//   /// VectorizationReport because the << operator of LoopSpawningReport returns
//   /// LoopAccessReport.
//   void emitAnalysis(const LoopAccessReport &Message) const {
//     emitAnalysisDiag(OrigLoop, *ORE, Message);
//   }
};

/// The LoopSpawning Pass.
struct LoopSpawningPass : public PassInfoMixin<LoopSpawningPass> {
  TapirTarget* tapirTarget;
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};
}

#endif // LLVM_TRANSFORMS_TAPIR_LOOPSPAWNING_H
