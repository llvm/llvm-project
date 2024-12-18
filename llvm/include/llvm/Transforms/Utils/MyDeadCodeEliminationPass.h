#ifndef LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H
#define LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include <unordered_set>
#include <string>

namespace llvm {

class MyDeadCodeEliminationPass : public PassInfoMixin<MyDeadCodeEliminationPass> {
public:
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
    void analyzeInstructionsIteratively(Function &F, FunctionAnalysisManager &AM);
    bool isInstructionDead(Instruction *Inst, const std::unordered_set<Instruction *> &potentialDeadInstructions);
    void printBasicBlockFeatures(const Instruction &I, const BasicBlock &BB);
    void printInstructionFeatures(const Instruction &I, const Function &F, const LoopInfo &LI);
    void printUsageFeatures(const Instruction &I, const std::unordered_set<Instruction *> &potentialDeadInstructions);
    std::string getInstructionPosition(const Instruction &I, const BasicBlock &BB);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

