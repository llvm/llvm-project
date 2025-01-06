//===-- RegUsageInfoCollector.cpp - Register Usage Information Collector --===//
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
/// This pass is simple MachineFunction pass which collects register usage
/// details by iterating through each physical registers and checking
/// MRI::isPhysRegUsed() then creates a RegMask based on this details.
/// The pass then stores this RegMask in PhysicalRegisterUsageInfo.cpp
///
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegUsageInfoCollector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterUsageInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "ip-regalloc"

STATISTIC(NumCSROpt,
          "Number of functions optimized for callee saved registers");

namespace {

class RegUsageInfoCollector {
  PhysicalRegisterUsageInfo &PRUI;

public:
  RegUsageInfoCollector(PhysicalRegisterUsageInfo &PRUI) : PRUI(PRUI) {}
  bool run(MachineFunction &MF);

  // Call getCalleeSaves and then also set the bits for subregs and
  // fully saved superregs.
  static void computeCalleeSavedRegs(BitVector &SavedRegs, MachineFunction &MF);
};

class RegUsageInfoCollectorLegacy : public MachineFunctionPass {
public:
  static char ID;
  RegUsageInfoCollectorLegacy() : MachineFunctionPass(ID) {
    initializeRegUsageInfoCollectorLegacyPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Register Usage Information Collector Pass";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PhysicalRegisterUsageInfoWrapperLegacy>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // end of anonymous namespace

char RegUsageInfoCollectorLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(RegUsageInfoCollectorLegacy, "RegUsageInfoCollector",
                      "Register Usage Information Collector", false, false)
INITIALIZE_PASS_DEPENDENCY(PhysicalRegisterUsageInfoWrapperLegacy)
INITIALIZE_PASS_END(RegUsageInfoCollectorLegacy, "RegUsageInfoCollector",
                    "Register Usage Information Collector", false, false)

FunctionPass *llvm::createRegUsageInfoCollector() {
  return new RegUsageInfoCollectorLegacy();
}

// TODO: Move to hook somwehere?

// Return true if it is useful to track the used registers for IPRA / no CSR
// optimizations. This is not useful for entry points, and computing the
// register usage information is expensive.
static bool isCallableFunction(const MachineFunction &MF) {
  switch (MF.getFunction().getCallingConv()) {
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_KERNEL:
    return false;
  default:
    return true;
  }
}

PreservedAnalyses
RegUsageInfoCollectorPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  Module &MFA = *MF.getFunction().getParent();
  auto *PRUI = MFAM.getResult<ModuleAnalysisManagerMachineFunctionProxy>(MF)
                   .getCachedResult<PhysicalRegisterUsageAnalysis>(MFA);
  assert(PRUI && "PhysicalRegisterUsageAnalysis not available");
  RegUsageInfoCollector(*PRUI).run(MF);
  return PreservedAnalyses::all();
}

bool RegUsageInfoCollectorLegacy::runOnMachineFunction(MachineFunction &MF) {
  PhysicalRegisterUsageInfo &PRUI =
      getAnalysis<PhysicalRegisterUsageInfoWrapperLegacy>().getPRUI();
  return RegUsageInfoCollector(PRUI).run(MF);
}

bool RegUsageInfoCollector::run(MachineFunction &MF) {
  MachineRegisterInfo *MRI = &MF.getRegInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  const TargetMachine &TM = MF.getTarget();

  LLVM_DEBUG(
      dbgs()
      << " -------------------- Register Usage Information Collector Pass"
      << " -------------------- \nFunction Name : " << MF.getName() << '\n');

  // Analyzing the register usage may be expensive on some targets.
  if (!isCallableFunction(MF)) {
    LLVM_DEBUG(dbgs() << "Not analyzing non-callable function\n");
    return false;
  }

  // If there are no callers, there's no point in computing more precise
  // register usage here.
  if (MF.getFunction().use_empty()) {
    LLVM_DEBUG(dbgs() << "Not analyzing function with no callers\n");
    return false;
  }

  std::vector<uint32_t> RegMask;

  // Compute the size of the bit vector to represent all the registers.
  // The bit vector is broken into 32-bit chunks, thus takes the ceil of
  // the number of registers divided by 32 for the size.
  unsigned RegMaskSize = MachineOperand::getRegMaskSize(TRI->getNumRegs());
  RegMask.resize(RegMaskSize, ~((uint32_t)0));

  const Function &F = MF.getFunction();

  PRUI.setTargetMachine(TM);

  LLVM_DEBUG(dbgs() << "Clobbered Registers: ");

  BitVector SavedRegs;
  computeCalleeSavedRegs(SavedRegs, MF);

  const BitVector &UsedPhysRegsMask = MRI->getUsedPhysRegsMask();
  auto SetRegAsDefined = [&RegMask] (unsigned Reg) {
    RegMask[Reg / 32] &= ~(1u << Reg % 32);
  };

  // Don't include $noreg in any regmasks.
  SetRegAsDefined(MCRegister::NoRegister);

  // Some targets can clobber registers "inside" a call, typically in
  // linker-generated code.
  for (const MCPhysReg Reg : TRI->getIntraCallClobberedRegs(&MF))
    for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
      SetRegAsDefined(*AI);

  // Scan all the physical registers. When a register is defined in the current
  // function set it and all the aliasing registers as defined in the regmask.
  // FIXME: Rewrite to use regunits.
  for (unsigned PReg = 1, PRegE = TRI->getNumRegs(); PReg < PRegE; ++PReg) {
    // Don't count registers that are saved and restored.
    if (SavedRegs.test(PReg))
      continue;
    // If a register is defined by an instruction mark it as defined together
    // with all it's unsaved aliases.
    if (!MRI->def_empty(PReg)) {
      for (MCRegAliasIterator AI(PReg, TRI, true); AI.isValid(); ++AI)
        if (!SavedRegs.test(*AI))
          SetRegAsDefined(*AI);
      continue;
    }
    // If a register is in the UsedPhysRegsMask set then mark it as defined.
    // All clobbered aliases will also be in the set, so we can skip setting
    // as defined all the aliases here.
    if (UsedPhysRegsMask.test(PReg))
      SetRegAsDefined(PReg);
  }

  if (TargetFrameLowering::isSafeForNoCSROpt(F) &&
      MF.getSubtarget().getFrameLowering()->isProfitableForNoCSROpt(F)) {
    ++NumCSROpt;
    LLVM_DEBUG(dbgs() << MF.getName()
                      << " function optimized for not having CSR.\n");
  }

  LLVM_DEBUG(
    for (unsigned PReg = 1, PRegE = TRI->getNumRegs(); PReg < PRegE; ++PReg) {
      if (MachineOperand::clobbersPhysReg(&(RegMask[0]), PReg))
        dbgs() << printReg(PReg, TRI) << " ";
    }

    dbgs() << " \n----------------------------------------\n";
  );

  PRUI.storeUpdateRegUsageInfo(F, RegMask);

  return false;
}

void RegUsageInfoCollector::
computeCalleeSavedRegs(BitVector &SavedRegs, MachineFunction &MF) {
  const TargetFrameLowering &TFI = *MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  // Target will return the set of registers that it saves/restores as needed.
  SavedRegs.clear();
  TFI.getCalleeSaves(MF, SavedRegs);
  if (SavedRegs.none())
    return;

  // Insert subregs.
  const MCPhysReg *CSRegs = TRI.getCalleeSavedRegs(&MF);
  for (unsigned i = 0; CSRegs[i]; ++i) {
    MCPhysReg Reg = CSRegs[i];
    if (SavedRegs.test(Reg)) {
      // Save subregisters
      for (MCPhysReg SR : TRI.subregs(Reg))
        SavedRegs.set(SR);
    }
  }
}
