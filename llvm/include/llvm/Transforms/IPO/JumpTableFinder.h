// #ifndef LLVM_ANALYSIS_JUMPTABLEFINDERPASS_H
// #define LLVM_ANALYSIS_JUMPTABLEFINDERPASS_H

// #include "llvm/IR/PassManager.h"
// #include "llvm/IR/Module.h"
// #include "llvm/Support/raw_ostream.h"
// #include <set>

// namespace llvm {

// class JumptableFinderPass : public PassInfoMixin<JumptableFinderPass> {
// public:
//     /// Main entry point for the pass. Analyzes the module to find and analyze jump tables.
//     PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
//     /// Implementation of the jump table finder.
//     void jumptableFinderImpl(Module &M);

//     /// Analyze a SwitchInst for potential jump table patterns.
//     void findJumpTableFromSwitch(SwitchInst *SI);

//     /// Analyze a GetElementPtrInst for jump table patterns.
//     void analyzeJumpTable(GetElementPtrInst *GEP);

//     /// Analyze the index computation of a jump table.
//     void analyzeIndex(Value *Index);

//     /// Find all potential targets for a jump table.
//     void findTargets(GetElementPtrInst *GEP, std::set<BasicBlock*> &Targets);

//     /// Check the density of a SwitchInst's cases to determine if it forms a jump table.
//     bool checkDensity(SwitchInst *SI);

//     /// Check if a GetElementPtrInst leads to an indirect branch.
//     bool leadsToIndirectBranch(GetElementPtrInst *GEP);
// };

// } // namespace llvm

// #endif // LLVM_ANALYSIS_JUMPTABLEFINDERPASS_H

#ifndef LLVM_TRANSFORMS_IPO_JUMPTABLEFINDER_H
#define LLVM_TRANSFORMS_IPO_JUMPTABLEFINDER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h" // For PassInfoMixin and PreservedAnalyses
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace llvm {

class JumptableFinderPass : public PassInfoMixin<JumptableFinderPass> {
public:
    // Entry point for the pass
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_IPO_JUMPTABLEFINDER_H
