//===- LoopPass.h - LoopPass class ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines LoopPass class. All loop optimization
// and transformation passes are derived from LoopPass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LOOP_PASS_H
#define LLVM_LOOP_PASS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"
#include "llvm/PassManagers.h"
#include "llvm/Function.h"

namespace llvm {

class LPPassManager;
class Function;
class PMStack;

class LoopPass : public Pass {

 public:
  explicit LoopPass(intptr_t pid) : Pass(pid) {}
  explicit LoopPass(void *pid) : Pass(pid) {}

  // runOnLoop - This method should be implemented by the subclass to perform
  // whatever action is necessary for the specified Loop.
  virtual bool runOnLoop (Loop *L, LPPassManager &LPM) = 0;
  virtual bool runOnFunctionBody (Function &F, LPPassManager &LPM) { 
    return false; 
  }

  // Initialization and finalization hooks.
  virtual bool doInitialization(Loop *L, LPPassManager &LPM) { 
    return false; 
  }

  // Finalization hook does not supply Loop because at this time
  // loop nest is completely different.
  virtual bool doFinalization() { return false; }

  // Check if this pass is suitable for the current LPPassManager, if
  // available. This pass P is not suitable for a LPPassManager if P
  // is not preserving higher level analysis info used by other
  // LPPassManager passes. In such case, pop LPPassManager from the
  // stack. This will force assignPassManager() to create new
  // LPPassManger as expected.
  void preparePassManager(PMStack &PMS);

  /// Assign pass manager to manager this pass
  virtual void assignPassManager(PMStack &PMS,
                                 PassManagerType PMT = PMT_LoopPassManager);

  ///  Return what kind of Pass Manager can manage this pass.
  virtual PassManagerType getPotentialPassManagerType() const {
    return PMT_LoopPassManager;
  }

  //===--------------------------------------------------------------------===//
  /// SimpleAnalysis - Provides simple interface to update analysis info
  /// maintained by various passes. Note, if required this interface can
  /// be extracted into a separate abstract class but it would require
  /// additional use of multiple inheritance in Pass class hierarchy, something
  /// we are trying to avoid.

  /// Each loop pass can override these simple analysis hooks to update
  /// desired analysis information.
  /// cloneBasicBlockAnalysis - Clone analysis info associated with basic block.
  virtual void cloneBasicBlockAnalysis(BasicBlock *F, BasicBlock *T, Loop *L) {}

  /// deletekAnalysisValue - Delete analysis info associated with value V.
  virtual void deleteAnalysisValue(Value *V, Loop *L) {}
};

class LPPassManager : public FunctionPass, public PMDataManager {

public:
  static char ID;
  explicit LPPassManager(int Depth);

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnFunction(Function &F);

  /// Pass Manager itself does not invalidate any analysis info.
  // LPPassManager needs LoopInfo. 
  void getAnalysisUsage(AnalysisUsage &Info) const; 
  
  virtual const char *getPassName() const {
    return "Loop Pass Manager";
  }
  
  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) {
    llvm::cerr << std::string(Offset*2, ' ') << "Loop Pass Manager\n";
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      Pass *P = getContainedPass(Index);
      P->dumpPassStructure(Offset + 1);
      dumpLastUses(P, Offset+1);
    }
  }
  
  Pass *getContainedPass(unsigned N) {
    assert ( N < PassVector.size() && "Pass number out of range!");
    Pass *FP = static_cast<Pass *>(PassVector[N]);
    return FP;
  }

  virtual PassManagerType getPassManagerType() const { 
    return PMT_LoopPassManager; 
  }

public:
  // Delete loop from the loop queue and loop nest (LoopInfo).
  void deleteLoopFromQueue(Loop *L);
  
  // Insert loop into the loop nest(LoopInfo) and loop queue(LQ).
  void insertLoop(Loop *L, Loop *ParentLoop);

  // Reoptimize this loop. LPPassManager will re-insert this loop into the
  // queue. This allows LoopPass to change loop nest for the loop. This
  // utility may send LPPassManager into infinite loops so use caution.
  void redoLoop(Loop *L);

  //===--------------------------------------------------------------------===//
  /// SimpleAnalysis - Provides simple interface to update analysis info
  /// maintained by various passes. Note, if required this interface can
  /// be extracted into a separate abstract class but it would require
  /// additional use of multiple inheritance in Pass class hierarchy, something
  /// we are trying to avoid.

  /// cloneBasicBlockSimpleAnalysis - Invoke cloneBasicBlockAnalysis hook for
  /// all passes that implement simple analysis interface.
  void cloneBasicBlockSimpleAnalysis(BasicBlock *From, BasicBlock *To, Loop *L);

  /// deleteSimpleAnalysisValue - Invoke deleteAnalysisValue hook for all passes
  /// that implement simple analysis interface.
  void deleteSimpleAnalysisValue(Value *V, Loop *L);
private:
  std::deque<Loop *> LQ;
  bool skipThisLoop;
  bool redoThisLoop;
  LoopInfo *LI;
  Loop *CurrentLoop;
};

} // End llvm namespace

#endif
