// MyDeadCodeEliminationPass.cpp
#include "llvm/Transforms/Utils/MyDeadCodeEliminationPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_set>
#include "llvm/IR/Module.h"


using namespace llvm;

PreservedAnalyses MyDeadCodeEliminationPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  errs() << "Starting MyDeadCodeEliminationPass\n";

  analyzeInstructionsIteratively(F, AM);

  return PreservedAnalyses::all();
}

void MyDeadCodeEliminationPass::analyzeInstructionsIteratively(
    Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  std::unordered_set<Instruction *> potentialDeadInstructions;
  bool foundNewDead;

  do {
    foundNewDead = false;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (potentialDeadInstructions.count(&I)) {
          continue;
        }

        // Check if the instruction is "likely dead"
        if (isInstructionDead(&I, potentialDeadInstructions)) {
          errs() << "Likely Dead Instruction: " << I << "\n";
          potentialDeadInstructions.insert(&I);
          foundNewDead = true;
        }

        // Print all features for this instruction
        errs() << "------------------------------------------\n";
        errs() << "Instruction: " << I << "\n";
        printInstructionFeatures(I, BB, F, LI);
        errs() << "------------------------------------------\n";
      }
    }
  } while (foundNewDead);
}

bool MyDeadCodeEliminationPass::isInstructionDead(
    Instruction *Inst,
    const std::unordered_set<Instruction *> &potentialDeadInstructions) {
  if (Inst->use_empty()) {
    return true;
  }

  for (const Use &U : Inst->uses()) {
    auto *User = dyn_cast<Instruction>(U.getUser());
    if (!User || potentialDeadInstructions.find(User) ==
                     potentialDeadInstructions.end()) {
      return false;
    }
  }

  return true;
}

void MyDeadCodeEliminationPass::printInstructionFeatures(const Instruction &I,
                                                         const BasicBlock &BB,
                                                         const Function &F,
                                                         const LoopInfo &LI) {
  // Direct Features
  errs() << "1. Opcode: " << I.getOpcodeName() << "\n";
  errs() << "2. Number of Operands: " << I.getNumOperands() << "\n";
  errs() << "3. Is Terminator: " << (I.isTerminator() ? "Yes" : "No") << "\n";
  errs() << "4. Is Volatile: "
         << (isa<LoadInst>(&I) || isa<StoreInst>(&I) ? "Yes" : "No") << "\n";
  errs() << "5. Has Metadata: " << (I.hasMetadata() ? "Yes" : "No") << "\n";

  if (isa<LoadInst>(&I)) {
    llvm::Align alignment = cast<LoadInst>(&I)->getAlign();
    errs() << "6. Alignment: " << alignment.value() << "\n";
  } else if (isa<StoreInst>(&I)) {
    llvm::Align alignment = cast<StoreInst>(&I)->getAlign();
    errs() << "6. Alignment: " << alignment.value() << "\n";
  } else {
    errs() << "6. Alignment: Not applicable\n";
  }

  
  /* Alignment alignment = cast<LoadInst>(&I)->getAlign();
  errs() << "6. Alignment: " << alignment.value() << "\n";*/
  /* errs() << "6. Alignment: "
         << (isa<LoadInst>(&I)
                 ? cast<LoadInst>(&I)->getAlign().valueOrOne().value()
                 : 0)
         << "\n";*/
  errs() << "7. Instruction Size: "
         << (I.getType()->isSized()
                 ? I.getModule()->getDataLayout().getTypeSizeInBits(I.getType())
                 : 0)
         << " bits\n";
  errs() << "8. Is PHI Node: " << (isa<PHINode>(&I) ? "Yes" : "No") << "\n";
  errs() << "9. Number of Users: " << I.getNumUses() << "\n";
  errs() << "10. Is Used in Loops: "
         << (LI.getLoopFor(I.getParent()) ? "Yes" : "No") << "\n";
  errs() << "11. Has Side Effects: " << (I.mayHaveSideEffects() ? "Yes" : "No")
         << "\n";

  // Basic Block-Level Features
  errs() << "12. Basic Block Predecessor Count: "
         << std::distance(pred_begin(&BB), pred_end(&BB)) << "\n";
  errs() << "13. Basic Block Successor Count: "
         << std::distance(succ_begin(&BB), succ_end(&BB)) << "\n";
  errs() << "14. Instruction Position in Basic Block: "
         << getInstructionPosition(I, BB) << "\n";
  errs() << "15. Basic Block Size: " << BB.size() << "\n";
  errs() << "16. Is Dominated by Entry: "
         << (F.getEntryBlock().getName() == BB.getName() ? "Yes" : "No")
         << "\n";
  errs() << "17. Loop Nesting Depth: " << (LI.getLoopDepth(&BB)) << "\n";

  // Function-Level Features
  errs() << "18. Function Argument Usage: "
         << (isUsingFunctionArguments(I, F) ? "Yes" : "No") << "\n";
  errs() << "19. Instruction Position in Function: "
         << getFunctionInstructionPosition(I, F) << "\n";
  errs() << "20. Function's Loop Count: " << LI.getLoopsInPreorder().size()
         << "\n";
}

bool MyDeadCodeEliminationPass::isUsingFunctionArguments(const Instruction &I,
                                                         const Function &F) {
  for (const Use &U : I.operands()) {
    if (isa<Argument>(U)) {
      return true;
    }
  }
  return false;
}

std::string
MyDeadCodeEliminationPass::getInstructionPosition(const Instruction &I,
                                                  const BasicBlock &BB) {
  size_t Position = 1;
  for (const Instruction &Inst : BB) {
    if (&Inst == &I) {
      break;
    }
    ++Position;
  }
  if (Position == 1) {
    return "First";
  } else if (&I == &BB.back()) {
    return "Last";
  } else {
    return "Intermediate (" + std::to_string(Position) + ")";
  }
}

std::string
MyDeadCodeEliminationPass::getFunctionInstructionPosition(const Instruction &I,
                                                          const Function &F) {
  size_t Position = 1;
  for (const BasicBlock &BB : F) {
    for (const Instruction &Inst : BB) {
      if (&Inst == &I) {
        return std::to_string(Position);
      }
      ++Position;
    }
  }
  return "Unknown";
}
