#include "llvm/Transforms/Utils/MyDeadCodeEliminationPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses MyDeadCodeEliminationPass::run(Function &F, FunctionAnalysisManager &AM) {
  errs() << "I'm here in my Pass" << "\n";

  // Call the helper function to analyze instructions
  analyzeInstructions(F);

  return PreservedAnalyses::all(); // Preserve analyses since the code isn't modified.
}

void MyDeadCodeEliminationPass::analyzeInstructions(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      errs() << "Instruction: " << I << "\n";

      // Check if the instruction is "dead" (not used anywhere)
      if (I.use_empty()) {
        errs() << "Potential dead instruction: " << I << "\n";
      }

      // Count instruction types
      if (isa<LoadInst>(&I)) {
        errs() << "Load Instruction\n";
      } else if (isa<StoreInst>(&I)) {
        errs() << "Store Instruction\n";
      }
    }
  }
}

