#ifndef LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H
#define LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include <unordered_set>

namespace llvm {

class MyDeadCodeEliminationPass : public PassInfoMixin<MyDeadCodeEliminationPass> {
public:
  // Main run method for the pass
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  // Helper function for iterative analysis
  void analyzeInstructionsIteratively(Function &F);

  // Helper function to check if an instruction is dead
  bool isInstructionDead(Instruction *Inst, const std::unordered_set<Instruction *> &potentialDeadInstructions);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

