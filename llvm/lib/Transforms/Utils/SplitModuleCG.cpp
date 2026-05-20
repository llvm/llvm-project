#include "llvm/Transforms/Utils/SplitModuleCG.h"

using namespace llvm;

#define DEBUG_TYPE "split-module-CG"

void SplitModuleCG::SplitModule(ModuleCreationCallback ModuleCallback,
                                const llvm::lto::Config &C) {
  // TODO: 1. Process the linkage of the GlobalValue; 2. Allocate the callgraph
  // to N partitions; 3.Invoke the cloneModule API to copy the N partitions to
  // obtain MParts.

}

SplitModuleCG::SplitModuleCG(Module &M,
                             const ModuleSummaryIndex &CombinedIndex,
                             unsigned LimitPartition)
    : M(M), CG(M), N(LimitPartition) {
  // TODO: The module is split based on the callgraph, and EntryFuncs stores
  // the root function of each callgraph.

  if (N == 0 || N > EntryFuncs.size()) {
    N = EntryFuncs.size();
  }
  N = N == 0 ? 1 : N;
}
