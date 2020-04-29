//===- MachineDebugify.cpp - Attach synthetic debug info to everything ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This pass attaches synthetic debug info to everything. It can be used
/// to create targeted tests for debug info preservation.
///
/// This isn't intended to have feature parity with Debugify.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/Debugify.h"

#define DEBUG_TYPE "mir-debugify"

using namespace llvm;

namespace {
bool applyDebugifyMetadataToMachineFunction(MachineModuleInfo &MMI,
                                            DIBuilder &DIB, Function &F) {
  MachineFunction *MaybeMF = MMI.getMachineFunction(F);
  if (!MaybeMF)
    return false;
  MachineFunction &MF = *MaybeMF;

  DISubprogram *SP = F.getSubprogram();
  assert(SP && "IR Debugify just created it?");

  LLVMContext &Ctx = F.getParent()->getContext();
  unsigned NextLine = SP->getLine();

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // This will likely emit line numbers beyond the end of the imagined
      // source function and into subsequent ones. We don't do anything about
      // that as it doesn't really matter to the compiler where the line is in
      // the imaginary source code.
      MI.setDebugLoc(DILocation::get(Ctx, NextLine++, 1, SP));
    }
  }

  return true;
}

/// ModulePass for attaching synthetic debug info to everything, used with the
/// legacy module pass manager.
struct DebugifyMachineModule : public ModulePass {
  bool runOnModule(Module &M) override {
    MachineModuleInfo &MMI =
        getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
    return applyDebugifyMetadata(
        M, M.functions(),
        "ModuleDebugify: ", [&](DIBuilder &DIB, Function &F) -> bool {
          return applyDebugifyMetadataToMachineFunction(MMI, DIB, F);
        });
  }

  DebugifyMachineModule() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.addPreserved<MachineModuleInfoWrapperPass>();
    AU.setPreservesCFG();
  }

  static char ID; // Pass identification.
};
char DebugifyMachineModule::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(DebugifyMachineModule, DEBUG_TYPE,
                      "Machine Debugify Module", false, false)
INITIALIZE_PASS_END(DebugifyMachineModule, DEBUG_TYPE,
                    "Machine Debugify Module", false, false)

ModulePass *llvm::createDebugifyMachineModulePass() {
  return new DebugifyMachineModule();
}
