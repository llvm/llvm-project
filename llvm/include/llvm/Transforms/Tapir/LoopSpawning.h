//===---- LoopSpawning.h ----------------------------------------*- C++ -*-===//
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
#include "llvm/Transforms/Tapir/TapirUtils.h"

#define LS_NAME "loop-spawning"

namespace llvm {

class AssumptionCache;
class DominatorTree;
class Function;
class LoopAccessInfo;
class LoopInfo;
class OptimizationRemarkEmitter;
class ScalarEvolution;
class TargetTransformInfo;

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
    TerminatorInst *TI = OrigLoop->getLoopLatch()->getTerminator();
    if (2 != TI->getNumSuccessors())
      return;
    ExitBlock = TI->getSuccessor(0);
    if (ExitBlock == OrigLoop->getHeader())
      ExitBlock = TI->getSuccessor(1);
  }

  virtual bool processLoop() = 0;

  virtual ~LoopOutline() {}

protected:
  PHINode* canonicalizeIVs(Type *Ty);
  Value* canonicalizeLoopLatch(PHINode *IV, Value *Limit);
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
