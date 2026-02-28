//===-- AMDGPUEarlyResourceCheck.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Check AMDGPU resource limits (LDS size) before register allocation so we
// fail fast instead of hanging in RA on impossible IR.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUEarlyResourceCheck.h"
#include "AMDGPU.h"
#include "AMDGPUMachineFunction.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-early-resource-check"

namespace {

class AMDGPUEarlyResourceCheckImpl {
public:
  bool run(MachineFunction &MF);
};

class AMDGPUEarlyResourceCheckLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUEarlyResourceCheckLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Early Resource Check";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // We may replace the function body on resource limit violation,
    // so we cannot claim to preserve all analyses.
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUEarlyResourceCheckLegacy, DEBUG_TYPE,
                "AMDGPU Early Resource Check", false, false)

char AMDGPUEarlyResourceCheckLegacy::ID = 0;

char &llvm::AMDGPUEarlyResourceCheckLegacyID =
    AMDGPUEarlyResourceCheckLegacy::ID;

bool AMDGPUEarlyResourceCheckLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  return AMDGPUEarlyResourceCheckImpl().run(MF);
}

PreservedAnalyses
AMDGPUEarlyResourceCheckPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  if (AMDGPUEarlyResourceCheckImpl().run(MF))
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}

bool AMDGPUEarlyResourceCheckImpl::run(MachineFunction &MF) {
  const AMDGPUMachineFunction *MFI = MF.getInfo<AMDGPUMachineFunction>();
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  unsigned Limit = STM.getAddressableLocalMemorySize();

  // Static LDS: from globals allocated during ISel. Same check as
  // AMDGPUAsmPrinter.cpp:1147-1153, but before RA instead of after.
  unsigned LDSUsed = MFI->getLDSSize();

  // Dynamic LDS: frontends that use zero-size LDS globals with runtime
  // allocation (e.g. Triton) communicate the expected size via function
  // attribute. The dynamic allocation is ON TOP of any static allocation,
  // so we add them.
  const Function &F = MF.getFunction();
  Attribute DynAttr = F.getFnAttribute("amdgpu-dynamic-lds-bytes");
  if (DynAttr.isStringAttribute()) {
    unsigned DynLDS = 0;
    DynAttr.getValueAsString().getAsInteger(0, DynLDS);
    LDSUsed += DynLDS;
  }

  if (LDSUsed > Limit) {
    LLVMContext &Ctx = F.getContext();
    DiagnosticInfoResourceLimit Diag(F, "local memory", LDSUsed, Limit,
                                     DS_Error);
    Ctx.diagnose(Diag);

    // Replace the function body with a minimal stub (single S_ENDPGM) so
    // that subsequent passes -- especially register allocation -- don't hang
    // trying to compile impossible IR. The diagnostic handler records the
    // error; the caller checks for it after the pass pipeline completes.
    const SIInstrInfo *TII = STM.getInstrInfo();
    while (!MF.empty())
      MF.erase(MF.begin());
    MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
    BuildMI(*MBB, MBB->end(), DebugLoc(), TII->get(AMDGPU::S_ENDPGM))
        .addImm(0);

    return true;
  }

  return false;
}
