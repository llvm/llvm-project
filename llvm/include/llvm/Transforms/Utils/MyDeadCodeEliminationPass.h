#ifndef LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H
#define LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include <string>
#include <unordered_set>

class MyDataSet;
namespace llvm {

class MyDeadCodeEliminationPass
    : public PassInfoMixin<MyDeadCodeEliminationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

private:
  void analyzeInstructionsIteratively(Function &F, FunctionAnalysisManager &AM);

  bool isInstructionDead(
      Instruction *Inst,
      const std::unordered_set<Instruction *> &potentialDeadInstructions);

  void printInstructionFeatures(const Instruction &I, const Function &F,
                                const LoopInfo &LI);

  void printBasicBlockFeatures(const Instruction &I, const BasicBlock &BB);

  void printUsageFeatures(
      const Instruction &I,
      const std::unordered_set<Instruction *> &potentialDeadInstructions);

  std::string getInstructionPosition(const Instruction &I,
                                     const BasicBlock &BB);

  int getInstructionDepth(
      const Instruction &I,
      const std::unordered_set<Instruction *> &potentialDeadInstructions);

  int getLoopNestingDepth(const LoopInfo &LI, const BasicBlock &BB);

  bool isDominatedByEntry(const DominatorTree &DT, const BasicBlock &BB);

  bool isInColdPath(const BasicBlock &BB, const BlockFrequencyInfo &BFI);

  void writeInstructionFeaturesToCSV(const Instruction &I, const Function &F,
                                     const LoopInfo &LI,
                                     const DominatorTree &DT,
                                     const BlockFrequencyInfo &BFI,
                                     const std::string &FilePath);

  std::string getFunctionInstructionPosition(const Instruction &I, const Function &F);

  bool isUsingFunctionArguments(const Instruction &I, const Function &F);

  void printInstructionFeatures(const Instruction &I, const BasicBlock &B,
                                const Function &F, const LoopInfo &L,
                                const DominatorTree &DT);
  void analyzeInstructionsIteratively(Function &F, FunctionAnalysisManager &AM,
                                      MyDataSet &dataSet);

  std::vector<std::string> collectInstructionFeatures(const Instruction &I, const BasicBlock &BB,
                                                      const Function &F, const LoopInfo &LI, 
                                                      const DominatorTree &DT);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_MYDEADCODEELIMINATIONPASS_H
