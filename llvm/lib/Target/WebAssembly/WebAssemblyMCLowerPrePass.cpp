//===-- WebAssemblyMCLowerPrePass.cpp - Prepare for MC lower --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Some information in MC lowering / asm printing gets generated as
/// instructions get emitted, but may be necessary at the start, such as for
/// .globaltype declarations. This pass collects this information.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-mclower-prepass"

namespace {
class WebAssemblyMCLowerPreLegacy final : public ModulePass {
  StringRef getPassName() const override {
    return "WebAssembly MC Lower Pre Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    ModulePass::getAnalysisUsage(AU);
  }

  bool runOnModule(Module &M) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyMCLowerPreLegacy() : ModulePass(ID) {}
};
} // end anonymous namespace

char WebAssemblyMCLowerPreLegacy::ID = 0;
INITIALIZE_PASS(WebAssemblyMCLowerPreLegacy, DEBUG_TYPE,
                "Collects information ahead of time for MC lowering", false,
                false)

ModulePass *llvm::createWebAssemblyMCLowerPreLegacyPass() {
  return new WebAssemblyMCLowerPreLegacy();
}

// NOTE: this is a ModulePass since we need to enforce that this code has run
// for all functions before AsmPrinter. If this way of doing things is ever
// suboptimal, we could opt to make it a MachineFunctionPass and instead use
// something like createBarrierNoopPass() to enforce ordering.
//
// The information stored here is essential for emitExternalDecls in the Wasm
// AsmPrinter
static void mcLower(Module &M, MachineModuleInfo &MMI,
                    llvm::function_ref<MachineFunction *(Function *)> GetMF) {
  MachineModuleInfoWasm &MMIW = MMI.getObjFileInfo<MachineModuleInfoWasm>();

  for (Function &F : M) {
    MachineFunction *MF = GetMF(&F);
    if (!MF)
      continue;

    LLVM_DEBUG(dbgs() << "********** MC Lower Pre Pass **********\n"
                         "********** Function: "
                      << MF->getName() << '\n');

    for (MachineBasicBlock &MBB : *MF) {
      for (auto &MI : MBB) {
        // FIXME: what should all be filtered out beyond these?
        if (MI.isDebugInstr() || MI.isInlineAsm())
          continue;
        for (MachineOperand &MO : MI.uses()) {
          if (MO.isSymbol()) {
            MMIW.MachineSymbolsUsed.insert(MO.getSymbolName());
          }
        }
      }
    }
  }
}

bool WebAssemblyMCLowerPreLegacy::runOnModule(Module &M) {
  auto *MMIWP = getAnalysisIfAvailable<MachineModuleInfoWrapperPass>();
  if (!MMIWP)
    return false;
  MachineModuleInfo &MMI = MMIWP->getMMI();
  mcLower(M, MMI, [MMIWP](Function *F) {
    return MMIWP->getMMI().getMachineFunction(*F);
  });
  return false;
}

PreservedAnalyses WebAssemblyMCLowerPrePass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  MachineModuleInfo &MMI = MAM.getResult<MachineModuleAnalysis>(M).getMMI();
  mcLower(M, MMI, [&](Function *F) {
    MachineFunctionAnalysis::Result *MFA =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M)
            .getManager()
            .getCachedResult<MachineFunctionAnalysis>(*F);
    return MFA ? &MFA->getMF() : nullptr;
  });
  return PreservedAnalyses::all();
}
