//===- NextSiliconWarnUnsupportedOMP.cpp ----------------------------------===//
//
// Emit a warning if a call to an OMP function not supported on NextSilicon
// device is encountered.
//
// Don't immediately issue an error, as the function call may not be
// scheduled to device.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/NextSiliconWarnUnsupportedOMP.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "warn-unsupported-omp"

static StringSet<> UnsupportedOMPFunctions{
    "omp_init_lock",
    "omp_set_lock",
    "omp_unset_lock",
    "omp_destroy_lock",
    "omp_get_max_active_levels",
    "omp_set_max_active_levels",
    "omp_get_supported_active_levels",
    "omp_get_num_places",
    "omp_get_place_num",
    "omp_get_place_num_procs",
    "omp_get_place_proc_ids",
    "omp_get_num_procs",
    "omp_set_affinity_format",
    "omp_get_affinity_format",
    "omp_display_affinity",
    "omp_capture_affinity",
};

static StringSet<> NonConformingOMPFunctions{
    "omp_get_max_threads",
    "omp_get_num_threads",
};

PreservedAnalyses
NextSiliconWarnUnsupportedOMPPass::run(Module &M, ModuleAnalysisManager &AM) {
  for (Function &F : M.functions()) {
    for (auto II = inst_begin(&F), IE = inst_end(&F); II != IE;) {
      Instruction *I = &*II++;
      CallInst *CallI = dyn_cast<CallInst>(I);
      InvokeInst *InvokeI = dyn_cast<InvokeInst>(I);

      CallBase *CB = CallI ? static_cast<CallBase *>(CallI)
                           : static_cast<CallBase *>(InvokeI);
      if (!CB)
        continue;

      Function *Callee = CB->getCalledFunction();
      if (!Callee)
        continue;

      if (UnsupportedOMPFunctions.contains(Callee->getName()))
        errs() << "Warning: In function '" << F.getName()
               << "': OMP function call to unsupported function '"
               << Callee->getName() << "'\n";
      else if (NonConformingOMPFunctions.contains(Callee->getName()))
        errs() << "Warning: In function '" << F.getName()
               << "': OMP function call to non-conforming function '"
               << Callee->getName() << "'\n";
    }
  }

  return PreservedAnalyses::all();
}
