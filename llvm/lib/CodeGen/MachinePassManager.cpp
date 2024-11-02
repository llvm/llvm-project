//===---------- MachinePassManager.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the pass management machinery for machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/PassManagerImpl.h"

using namespace llvm;

namespace llvm {

AnalysisKey FunctionAnalysisManagerMachineFunctionProxy::Key;

template class AnalysisManager<MachineFunction>;
template class PassManager<MachineFunction>;
template class InnerAnalysisManagerProxy<MachineFunctionAnalysisManager,
                                         Module>;
template class OuterAnalysisManagerProxy<ModuleAnalysisManager,
                                         MachineFunction>;

bool FunctionAnalysisManagerMachineFunctionProxy::Result::invalidate(
    MachineFunction &IR, const PreservedAnalyses &PA,
    MachineFunctionAnalysisManager::Invalidator &Inv) {
  // MachineFunction passes should not invalidate Function analyses.
  // TODO: verify that PA doesn't invalidate Function analyses.
  return false;
}

template <>
bool MachineFunctionAnalysisManagerModuleProxy::Result::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &Inv) {
  // If literally everything is preserved, we're done.
  if (PA.areAllPreserved())
    return false; // This is still a valid proxy.

  // If this proxy isn't marked as preserved, then even if the result remains
  // valid, the key itself may no longer be valid, so we clear everything.
  //
  // Note that in order to preserve this proxy, a module pass must ensure that
  // the MFAM has been completely updated to handle the deletion of functions.
  // Specifically, any MFAM-cached results for those functions need to have been
  // forcibly cleared. When preserved, this proxy will only invalidate results
  // cached on functions *still in the module* at the end of the module pass.
  auto PAC = PA.getChecker<MachineFunctionAnalysisManagerModuleProxy>();
  if (!PAC.preserved() && !PAC.preservedSet<AllAnalysesOn<Module>>()) {
    InnerAM->clear();
    return true;
  }

  // FIXME: be more precise, see
  // FunctionAnalysisManagerModuleProxy::Result::invalidate.
  if (!PA.allAnalysesInSetPreserved<AllAnalysesOn<MachineFunction>>()) {
    InnerAM->clear();
    return true;
  }

  // Return false to indicate that this result is still a valid proxy.
  return false;
}

PreservedAnalyses
ModuleToMachineFunctionPassAdaptor::run(Module &M, ModuleAnalysisManager &AM) {
  auto &MMI = AM.getResult<MachineModuleAnalysis>(M).getMMI();
  MachineFunctionAnalysisManager &MFAM =
      AM.getResult<MachineFunctionAnalysisManagerModuleProxy>(M).getManager();
  PassInstrumentation PI = AM.getResult<PassInstrumentationAnalysis>(M);
  PreservedAnalyses PA = PreservedAnalyses::all();
  for (Function &F : M) {
    // Do not codegen any 'available_externally' functions at all, they have
    // definitions outside the translation unit.
    if (F.hasAvailableExternallyLinkage())
      continue;

    MachineFunction &MF = MMI.getOrCreateMachineFunction(F);

    if (!PI.runBeforePass<MachineFunction>(*Pass, MF))
      continue;
    PreservedAnalyses PassPA = Pass->run(MF, MFAM);
    if (MMI.getMachineFunction(F)) {
      MFAM.invalidate(MF, PassPA);
      PI.runAfterPass(*Pass, MF, PassPA);
    } else {
      MFAM.clear(MF, F.getName());
      PI.runAfterPassInvalidated<MachineFunction>(*Pass, PassPA);
    }
    PA.intersect(std::move(PassPA));
  }

  return PA;
}

void ModuleToMachineFunctionPassAdaptor::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  OS << "machine-function(";
  Pass->printPipeline(OS, MapClassName2PassName);
  OS << ')';
}

template <>
PreservedAnalyses
PassManager<MachineFunction>::run(MachineFunction &MF,
                                  AnalysisManager<MachineFunction> &MFAM) {
  PassInstrumentation PI = MFAM.getResult<PassInstrumentationAnalysis>(MF);
  Function &F = MF.getFunction();
  MachineModuleInfo &MMI =
      MFAM.getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
          .getCachedResult<MachineModuleAnalysis>(*F.getParent())
          ->getMMI();
  PreservedAnalyses PA = PreservedAnalyses::all();
  for (auto &Pass : Passes) {
    if (!PI.runBeforePass<MachineFunction>(*Pass, MF))
      continue;

    PreservedAnalyses PassPA = Pass->run(MF, MFAM);
    if (MMI.getMachineFunction(F)) {
      MFAM.invalidate(MF, PassPA);
      PI.runAfterPass(*Pass, MF, PassPA);
    } else {
      MFAM.clear(MF, F.getName());
      PI.runAfterPassInvalidated<MachineFunction>(*Pass, PassPA);
    }
    PA.intersect(std::move(PassPA));
  }
  return PA;
}

} // namespace llvm
