#include "llvm/Transforms/Utils/MyDeadCodeEliminationPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/LoopInfo.h"
#include <unordered_set>

using namespace llvm;

PreservedAnalyses MyDeadCodeEliminationPass::run(Function &F, FunctionAnalysisManager &AM) {
    errs() << "Starting MyDeadCodeEliminationPass\n";

    analyzeInstructionsIteratively(F, AM);

    return PreservedAnalyses::all();
}

void MyDeadCodeEliminationPass::analyzeInstructionsIteratively(Function &F, FunctionAnalysisManager &AM) {
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
                printBasicBlockFeatures(I, BB);
                printInstructionFeatures(I, F, LI);
                printUsageFeatures(I, potentialDeadInstructions);
                errs() << "------------------------------------------\n";
            }
        }
    } while (foundNewDead);
}

bool MyDeadCodeEliminationPass::isInstructionDead(Instruction *Inst, const std::unordered_set<Instruction *> &potentialDeadInstructions) {
    if (Inst->use_empty()) {
        return true;
    }

    for (const Use &U : Inst->uses()) {
        auto *User = dyn_cast<Instruction>(U.getUser());
        if (!User || potentialDeadInstructions.find(User) == potentialDeadInstructions.end()) {
            return false;
        }
    }

    return true;
}

void MyDeadCodeEliminationPass::printBasicBlockFeatures(const Instruction &I, const BasicBlock &BB) {
    errs() << "Parent Basic Block: " << BB.getName() << "\n";
    errs() << "  Number of Predecessors: " << std::distance(pred_begin(&BB), pred_end(&BB)) << "\n";
    errs() << "  Number of Successors: " << std::distance(succ_begin(&BB), succ_end(&BB)) << "\n";
    errs() << "  Position in Basic Block: " << getInstructionPosition(I, BB) << "\n";
    errs() << "  Total Instructions in Basic Block: " << BB.size() << "\n";
}

void MyDeadCodeEliminationPass::printInstructionFeatures(const Instruction &I, const Function &F, const LoopInfo &LI) {
    errs() << "Instruction Type: " << I.getOpcodeName() << "\n";
    errs() << "Operand Count: " << I.getNumOperands() << "\n";
    errs() << "Is Terminator: " << (I.isTerminator() ? "Yes" : "No") << "\n";
    errs() << "Is Memory Related: " << (isa<LoadInst>(&I) || isa<StoreInst>(&I) ? "Yes" : "No") << "\n";

    if (LI.getLoopFor(I.getParent())) {
        errs() << "Part of a Loop: Yes\n";
    } else {
        errs() << "Part of a Loop: No\n";
    }
}


void MyDeadCodeEliminationPass::printUsageFeatures(const Instruction &I, const std::unordered_set<Instruction *> &potentialDeadInstructions) {
    errs() << "Uses: ";
    for (const Use &U : I.uses()) {
        const auto *User = dyn_cast<Instruction>(U.getUser());
        if (User) {
            errs() << User->getOpcodeName() << " ";
            // Fix: Cast to match the type stored in the set
            if (potentialDeadInstructions.count(const_cast<Instruction *>(User))) {
                errs() << "(likely dead) ";
            }
        }
    }
    errs() << "\n";
    errs() << "Number of Users: " << I.getNumUses() << "\n";
}


std::string MyDeadCodeEliminationPass::getInstructionPosition(const Instruction &I, const BasicBlock &BB) {
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

