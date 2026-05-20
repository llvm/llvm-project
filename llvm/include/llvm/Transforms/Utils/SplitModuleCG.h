#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H

#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/LTO/Config.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {
/// Splits the module M into N linkable partitions. The function ModuleCallback
/// is called N times passing each individual partition as the MPart argument.
class SplitModuleCG {
public:
  using ModuleCreationCallback =
      function_ref<void(std::unique_ptr<Module> MPart, unsigned PartitionId)>;
  SplitModuleCG(Module &M,
                const ModuleSummaryIndex &CombinedIndex,
                unsigned LimitPartition = 0);
  void SplitModule(ModuleCreationCallback ModuleCallback,
                   const llvm::lto::Config &C);

  unsigned getPartitionNum() { return N; }

  private:
  unsigned N;
  Module &M;
  CallGraph CG;
  DenseSet<const Function *> EntryFuncs;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULECG_H
