#include "llvm/Transforms/Utils/MyDeadCodeEliminationPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>

using namespace llvm;

PreservedAnalyses MyDeadCodeEliminationPass::run(Function &F, FunctionAnalysisManager &AM) {
  errs() << "I'm here in my Pass" << "\n";

  // Call the helper function to iteratively analyze instructions
  analyzeInstructionsIteratively(F);

  return PreservedAnalyses::all(); // Preserve analyses since the code isn't modified.
}

void MyDeadCodeEliminationPass::analyzeInstructionsIteratively(Function &F) {
  std::unordered_set<Instruction *> potentialDeadInstructions; // To track potential dead instructions
  bool foundNewDead; // Flag to track if we find new dead instructions in an iteration

  do {
    foundNewDead = false; // Reset the flag at the start of each iteration

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        // Skip if already identified as dead
        if (potentialDeadInstructions.count(&I)) {
          continue;
        }

        // Check if the instruction is "dead" (not used anywhere or used only by dead instructions)
        if (isInstructionDead(&I, potentialDeadInstructions)) {
          errs() << "Potential dead instruction: " << I << "\n";
          potentialDeadInstructions.insert(&I);
          foundNewDead = true; // Mark that we found a new dead instruction
        }
      }
    }
  } while (foundNewDead); // Continue until no new dead instructions are found
}

bool MyDeadCodeEliminationPass::isInstructionDead(Instruction *Inst, const std::unordered_set<Instruction *> &potentialDeadInstructions) {
  // Check if the instruction's result is not used, or all users are in the potentialDeadInstructions set
  if (Inst->use_empty()) {
    return true; // No users, definitely dead
  }

  for (const Use &U : Inst->uses()) {
    auto *User = dyn_cast<Instruction>(U.getUser());
    if (!User || potentialDeadInstructions.find(User) == potentialDeadInstructions.end()) {
      return false; // Found a user that is not "dead"
    }
  }

  return true; // All users are "dead"
}

