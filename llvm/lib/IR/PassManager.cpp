//===- PassManager.cpp - Infrastructure for managing & running IR passes --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

// Explicit template instantiations and specialization defininitions for core
// template typedefs.
namespace llvm {
template class AllAnalysesOn<Module>;
template class AllAnalysesOn<Function>;
template class PassManager<Module>;
template class PassManager<Function>;
template class AnalysisManager<Module>;
template class AnalysisManager<Function>;
template class InnerAnalysisManagerProxy<FunctionAnalysisManager, Module>;
template class OuterAnalysisManagerProxy<ModuleAnalysisManager, Function>;

template <>
bool FunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &Inv) {
  // If this proxy isn't marked as preserved, then even if the result remains
  // valid, the key itself may no longer be valid, so we clear everything.
  //
  // Note that in order to preserve this proxy, a module pass must ensure that
  // the FAM has been completely updated to handle the deletion of functions.
  // Specifically, any FAM-cached results for those functions need to have been
  // forcibly cleared. When preserved, this proxy will only invalidate results
  // cached on functions *still in the module* at the end of the module pass.
  if (!PA.preserved(FunctionAnalysisManagerModuleProxy::ID())) {
    InnerAM->clear();
    return true;
  }

  // Otherwise propagate the invalidation event to all the remaining IR units.
  for (Function &F : M)
    InnerAM->invalidate(F, PA);

  // Return false to indicate that this result is still a valid proxy.
  return false;
}
}

AnalysisKey PreservedAnalyses::AllAnalysesKey;
