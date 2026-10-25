#define DEBUG_TYPE "bsize-invariance-analysis"

#include "llvm/Transforms/BlockSizeInvarianceAnalysisPass.h"

#include <cxxabi.h>

using namespace llvm;

inline std::string demangle(const char* name) 
{
  int status = -1; 

  std::unique_ptr<char, void(*)(void*)> res {
      abi::__cxa_demangle(name, NULL, NULL, &status), std::free };
  return (status == 0) ? res.get() : std::string(name);
}


bool BlockSizeInvarianceAnalysisPass::runOnFunction(Function &F) {
  // BSizeDependenceValue::testBSizeDependenceValue();
  // BSizeGPUState::testBSizeGPUState();

  LLVM_DEBUG(errs() << "-------------- Computing Block-size Invariance ------------------\n");
  errs() << "Function: " << demangle(F.getName().data()) << "\n";

  std::set<const Instruction*> dependentAccesses;
  std::set<const Instruction*> syncthreads;
  auto &DomTree = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  for (int i = 0; i < 3; i++) {
    BlockSizeInvarianceAnalysis BDA(&F, &DomTree, i);
    BSizeGPUState st = BDA.BuildInitialState();
    BDA.BuildAnalysisInfo(st);
    auto depSet = BDA.getBlockSizeDependentAccesses();
    auto syncSet = BDA.getSyncThreads();
    dependentAccesses.insert(depSet.begin(), depSet.end());
    syncthreads.insert(syncSet.begin(), syncSet.end());
  }

  // If block-size dependent accesses is empty, print block-size invariance.
  if (dependentAccesses.empty() && syncthreads.empty()) {
    errs() << "Function " << demangle(F.getName().data()) << " is block-size independent!\n";
    return false;
  }
  // Print block-size dependent accesses found by the analysis.
  errs() << "  Block-size dependent accesses: #" 
      << dependentAccesses.size() << "\n";
  for (auto it = dependentAccesses.begin(), ite = dependentAccesses.end();
           it != ite; ++it) {
    errs() << "  -- ";
    (*it)->getDebugLoc().print(errs());
    errs() << "\n";
  }
  errs() << "\n";

  return false;
}

char BlockSizeInvarianceAnalysisPass::ID = 0;
static RegisterPass<BlockSizeInvarianceAnalysisPass>
Y("bsize-invariance-analysis", "Pass to check block-size invariance of GPU kernels.");

FunctionPass* createBlockSizeInvarianceAnalysisPass() {
  return new BlockSizeInvarianceAnalysisPass();
}
