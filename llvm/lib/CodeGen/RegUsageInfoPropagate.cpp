//=--- RegUsageInfoPropagate.cpp - Register Usage Informartion Propagation --=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This pass is required to take advantage of the interprocedural register
/// allocation infrastructure.
///
/// This pass iterates through MachineInstrs in a given MachineFunction and at
/// each callsite queries RegisterUsageInfo for RegMask (calculated based on
/// actual register allocation) of the callee function, if the RegMask detail
/// is available then this pass will update the RegMask of the call instruction.
/// This updated RegMask will be used by the register allocator while allocating
/// the current MachineFunction.
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegUsageInfoPropagate.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterUsageInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ip-regalloc"

#define RUIP_NAME "Register Usage Information Propagation"

namespace {

class RegUsageInfoPropagation {
public:
  explicit RegUsageInfoPropagation(PhysicalRegisterUsageInfo *PRUI)
      : PRUI(PRUI) {}

  bool run(MachineFunction &MF);

private:
  PhysicalRegisterUsageInfo *PRUI;

  static void setRegMask(MachineInstr &MI, ArrayRef<uint32_t> RegMask) {
    assert(RegMask.size() ==
           MachineOperand::getRegMaskSize(MI.getParent()->getParent()
                                          ->getRegInfo().getTargetRegisterInfo()
                                          ->getNumRegs())
           && "expected register mask size");
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask())
        MO.setRegMask(RegMask.data());
    }
  }
};

class RegUsageInfoPropagationLegacy : public MachineFunctionPass {
public:
  static char ID;
  RegUsageInfoPropagationLegacy() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeRegUsageInfoPropagationLegacyPass(Registry);
  }

  StringRef getPassName() const override { return RUIP_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PhysicalRegisterUsageInfoWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end of anonymous namespace

INITIALIZE_PASS_BEGIN(RegUsageInfoPropagationLegacy, "reg-usage-propagation",
                      RUIP_NAME, false, false)
INITIALIZE_PASS_DEPENDENCY(PhysicalRegisterUsageInfoWrapperLegacy)
INITIALIZE_PASS_END(RegUsageInfoPropagationLegacy, "reg-usage-propagation",
                    RUIP_NAME, false, false)

char RegUsageInfoPropagationLegacy::ID = 0;

// Assumes call instructions have a single reference to a function.
static const Function *findCalledFunction(const Module &M,
                                          const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isGlobal())
      return dyn_cast<const Function>(MO.getGlobal());

    if (MO.isSymbol())
      return M.getFunction(MO.getSymbolName());
  }

  return nullptr;
}

bool RegUsageInfoPropagationLegacy::runOnMachineFunction(MachineFunction &MF) {
  PhysicalRegisterUsageInfo *PRUI =
      &getAnalysis<PhysicalRegisterUsageInfoWrapperLegacy>().getPRUI();

  RegUsageInfoPropagation RUIP(PRUI);
  return RUIP.run(MF);
}

PreservedAnalyses
RegUsageInfoPropagationPass::run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM) {
  Module &MFA = *MF.getFunction().getParent();
  auto *PRUI = MFAM.getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
                   .getCachedResult<PhysicalRegisterUsageAnalysis>(MFA);
  assert(PRUI && "PhysicalRegisterUsageAnalysis not available");
  RegUsageInfoPropagation(PRUI).run(MF);
  return PreservedAnalyses::all();
}

bool RegUsageInfoPropagation::run(MachineFunction &MF) {
  const Module &M = *MF.getFunction().getParent();

  LLVM_DEBUG(dbgs() << " ++++++++++++++++++++ " << RUIP_NAME
                    << " ++++++++++++++++++++  \n");
  LLVM_DEBUG(dbgs() << "MachineFunction : " << MF.getName() << "\n");

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasCalls() && !MFI.hasTailCall())
    return false;

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!MI.isCall())
        continue;
      LLVM_DEBUG(
          dbgs()
          << "Call Instruction Before Register Usage Info Propagation : \n"
          << MI << "\n");

      auto UpdateRegMask = [&](const Function &F) {
        const ArrayRef<uint32_t> RegMask = PRUI->getRegUsageInfo(F);
        if (RegMask.empty())
          return;
        setRegMask(MI, RegMask);
        Changed = true;
      };

      if (const Function *F = findCalledFunction(M, MI)) {
        if (F->isDefinitionExact()) {
          UpdateRegMask(*F);
        } else {
          LLVM_DEBUG(dbgs() << "Function definition is not exact\n");
        }
      } else {
        LLVM_DEBUG(dbgs() << "Failed to find call target function\n");
      }

      LLVM_DEBUG(
          dbgs()
          << "Call Instruction After Register Usage Info Propagation : \n"
          << MI << '\n');
    }
  }

  LLVM_DEBUG(
      dbgs() << " +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                "++++++ \n");
  return Changed;
}

FunctionPass *llvm::createRegUsageInfoPropPass() {
  return new RegUsageInfoPropagationLegacy();
}
