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
template class AllAnalysesOn<MachineFunction>;
template class AnalysisManager<MachineFunction>;
template class PassManager<MachineFunction>;

Error MachineFunctionPassManager::run(Module &M,
                                      MachineFunctionAnalysisManager &MFAM) {
  // MachineModuleAnalysis is a module analysis pass that is never invalidated
  // because we don't run any module pass in codegen pipeline. This is very
  // important because the codegen state is stored in MMI which is the analysis
  // result of MachineModuleAnalysis. MMI should not be recomputed.
  auto &MMI = MFAM.getResult<MachineModuleAnalysis>(M);

  (void)RequireCodeGenSCCOrder;
  assert(!RequireCodeGenSCCOrder && "not implemented");

  // Add a PIC to verify machine functions.
  if (VerifyMachineFunction) {
    PassInstrumentation PI = MFAM.getResult<PassInstrumentationAnalysis>(M);

    // No need to pop this callback later since MIR pipeline is flat which means
    // current pipeline is the top-level pipeline. Callbacks are not used after
    // current pipeline.
    PI.pushBeforeNonSkippedPassCallback([&MFAM](StringRef PassID, Any IR) {
      assert(llvm::any_cast<const MachineFunction *>(&IR));
      const MachineFunction *MF = llvm::any_cast<const MachineFunction *>(IR);
      assert(MF && "Machine function should be valid for printing");
      std::string Banner = std::string("After ") + std::string(PassID);
      verifyMachineFunction(&MFAM, Banner, *MF);
    });
  }

  for (auto &F : InitializationFuncs) {
    if (auto Err = F(M, MFAM))
      return Err;
  }

  unsigned Idx = 0;
  size_t Size = Passes.size();
  do {
    // Run machine module passes
    for (; MachineModulePasses.count(Idx) && Idx != Size; ++Idx) {
      if (auto Err = MachineModulePasses.at(Idx)(M, MFAM))
        return Err;
    }

    // Finish running all passes.
    if (Idx == Size)
      break;

    // Run machine function passes

    // Get index range of machine function passes.
    unsigned Begin = Idx;
    for (; !MachineModulePasses.count(Idx) && Idx != Size; ++Idx)
      ;

    for (Function &F : M) {
      // Do not codegen any 'available_externally' functions at all, they have
      // definitions outside the translation unit.
      if (F.hasAvailableExternallyLinkage())
        continue;

      MachineFunction &MF = MMI.getOrCreateMachineFunction(F);
      PassInstrumentation PI = MFAM.getResult<PassInstrumentationAnalysis>(MF);

      for (unsigned I = Begin, E = Idx; I != E; ++I) {
        auto *P = Passes[I].get();

        if (!PI.runBeforePass<MachineFunction>(*P, MF))
          continue;

        // TODO: EmitSizeRemarks
        PreservedAnalyses PassPA = P->run(MF, MFAM);
        MFAM.invalidate(MF, PassPA);
        PI.runAfterPass(*P, MF, PassPA);
      }
    }
  } while (true);

  for (auto &F : FinalizationFuncs) {
    if (auto Err = F(M, MFAM))
      return Err;
  }

  return Error::success();
}

} // namespace llvm
