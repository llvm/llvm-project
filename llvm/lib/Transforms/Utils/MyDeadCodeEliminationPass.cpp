#include "llvm/Transforms/Utils/MyDeadCodeEliminationPass.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream> // For CSV file handling
#include <unordered_set>


// Class to handle CSV file operations
class MyDataSet {
private:
  std::ofstream outFile;

public:
  // Constructor opens the file and writes the header row
  MyDataSet(const std::string &fileName) {
    outFile.open(fileName);
    if (outFile.is_open()) {
      outFile << "Opcode,Number of Operands,Is Terminator,Is Volatile,Has "
                 "Metadata,";
      outFile << "Alignment,Instruction Size,Is PHI Node,Number of Users,Is "
                 "Used in Loops,";
      outFile << "Has Side Effects,Is Constant,Basic Block Predecessor "
                 "Count,Basic Block Successor Count,";
      outFile << "Instruction Position in Basic Block,Is Entry Block,Is "
                 "Dominated by Entry,";
      outFile << "Loop Nesting Depth,Function Argument Usage,Instruction "
                 "Position in Function,";
      outFile << "Function's Loop Count,Function Call Depth,Module Size,Is in "
                 "Cold Path,Call Graph Features\n";
    } else {
      llvm::errs() << "Failed to open file: " << fileName << "\n";
    }
  }

  // Method to write a single row
  void WriteRow(const std::vector<std::string> &features) {
    if (!outFile.is_open())
      return;
    for (size_t i = 0; i < features.size(); ++i) {
      outFile << features[i];
      if (i != features.size() - 1)
        outFile << ",";
    }
    outFile << "\n";
  }

  // Destructor closes the file
  ~MyDataSet() {
    if (outFile.is_open()) {
      outFile.close();
    }
  }
};

using namespace llvm;

PreservedAnalyses MyDeadCodeEliminationPass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  errs() << "Starting MyDeadCodeEliminationPass\n";

  // Create an instance of MyDataSet to manage the CSV file
  MyDataSet dataSet("InstructionFeatures.csv");

  analyzeInstructionsIteratively(F, AM, dataSet);

  return PreservedAnalyses::all();
}

void MyDeadCodeEliminationPass::analyzeInstructionsIteratively(
    Function &F, FunctionAnalysisManager &AM, MyDataSet &dataSet) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
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

        // Collect features and write to CSV
        std::vector<std::string> features =
            collectInstructionFeatures(I, BB, F, LI, DT);
        dataSet.WriteRow(features);

        // Print all features for this instruction
        errs() << "------------------------------------------\n";
        errs() << "Instruction: " << I << "\n";
        for (size_t i = 0; i < features.size(); ++i) {
          errs() << (i + 1) << ". " << features[i] << "\n";
        }
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

std::vector<std::string> MyDeadCodeEliminationPass::collectInstructionFeatures(
    const Instruction &I, const BasicBlock &BB, const Function &F,
    const LoopInfo &LI, const DominatorTree &DT) {

  std::vector<std::string> features;

  // Direct Features
  features.push_back(I.getOpcodeName());
  features.push_back(std::to_string(I.getNumOperands()));
  features.push_back(I.isTerminator() ? "Yes" : "No");
  features.push_back((isa<LoadInst>(&I) || isa<StoreInst>(&I)) ? "Yes" : "No");
  features.push_back(I.hasMetadata() ? "Yes" : "No");

  if (isa<LoadInst>(&I)) {
    llvm::Align alignment = cast<LoadInst>(&I)->getAlign();
    features.push_back(std::to_string(alignment.value()));
  } else if (isa<StoreInst>(&I)) {
    llvm::Align alignment = cast<StoreInst>(&I)->getAlign();
    features.push_back(std::to_string(alignment.value()));
  } else {
    features.push_back("Not applicable");
  }

  features.push_back(
      I.getType()->isSized()
          ? std::to_string(
                I.getModule()->getDataLayout().getTypeSizeInBits(I.getType()))
          : "0");
  features.push_back(isa<PHINode>(&I) ? "Yes" : "No");
  features.push_back(std::to_string(I.getNumUses()));
  features.push_back(LI.getLoopFor(I.getParent()) ? "Yes" : "No");
  features.push_back(I.mayHaveSideEffects() ? "Yes" : "No");
  features.push_back(isa<Constant>(&I) ? "Yes" : "No");

  // Basic Block-Level Features
  features.push_back(
      std::to_string(std::distance(pred_begin(&BB), pred_end(&BB))));
  features.push_back(
      std::to_string(std::distance(succ_begin(&BB), succ_end(&BB))));
  features.push_back(getInstructionPosition(I, BB));
  features.push_back(F.getEntryBlock().getName() == BB.getName() ? "Yes"
                                                                 : "No");
  features.push_back(DT.dominates(&F.getEntryBlock(), &BB) ? "Yes" : "No");
  features.push_back(std::to_string(LI.getLoopDepth(&BB)));

  // Function-Level Features
  features.push_back(isUsingFunctionArguments(I, F) ? "Yes" : "No");
  features.push_back(getFunctionInstructionPosition(I, F));
  features.push_back(std::to_string(LI.getLoopsInPreorder().size()));
  features.push_back("0"); // Placeholder for Function Call Depth

  // Module-Level Features
  features.push_back(std::to_string(F.getParent()->size()));
  features.push_back("Unknown");     // Placeholder for Is in Cold Path
  features.push_back("Unavailable"); // Placeholder for Call Graph Features

  return features;
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
