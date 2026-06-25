//===-- GCNMFMAAccumMacChain.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Convert non-mac MFMA accumulate chains to mac form with tied-def on the
/// accumulator vreg. The SSA machine scheduler leaves ping-pong early-clobber
/// MFMAs (dest != acc vreg); this pass folds each accumulate step onto the
/// acc operand and replaces the discarded dest vreg so tile pressure matches
/// the mac+tied-def form seen without the SSA scheduler.
//
//===----------------------------------------------------------------------===//

#include "GCNMFMAAccumMacChain.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-mfma-accum-mac-chain"

static cl::opt<bool>
    EnableMFMAAccumMacChain("amdgpu-mfma-accum-mac-chain", cl::init(true),
                            cl::Hidden,
                            cl::desc("Convert MFMA accumulate chains to mac "
                                     "form with tied-def before RA"));

STATISTIC(NumMFMAAccumMacCandidates,
          "Number of non-mac MFMAs with vreg accumulator");
STATISTIC(NumMFMAAccumMacConverted,
          "Number of MFMAs converted to mac tied-def form");
STATISTIC(NumMFMAAccumMacDestRegsFolded,
          "Number of MFMA dest vregs folded into accumulator vregs");
STATISTIC(NumMFMAAccumMacSkipped,
          "Number of MFMAs skipped (dest feeds non-chain uses)");

static bool convertMFMAToMac(const SIInstrInfo &TII, MachineRegisterInfo &MRI,
                             MachineInstr &MI) {
  if (!TII.isMFMA(MI))
    return false;

  // Already mac form.
  if (AMDGPU::getMFMAEarlyClobberOp(MI.getOpcode()) != -1)
    return false;

  int MacOpc = AMDGPU::getMFMAMacOp(MI.getOpcode());
  if (MacOpc == -1)
    return false;

  MachineOperand *Src2 = TII.getNamedOperand(MI, AMDGPU::OpName::src2);
  if (!Src2 || !Src2->isReg())
    return false;

  Register Acc = Src2->getReg();
  if (!Acc.isVirtual())
    return false;

  ++NumMFMAAccumMacCandidates;

  MachineOperand &Dest = MI.getOperand(0);
  Register OldDest = Dest.getReg();
  if (!OldDest.isVirtual())
    return false;

  if (OldDest != Acc && !TII.canFoldMFMAMacDestIntoAcc(MRI, OldDest, Acc, MI)) {
    ++NumMFMAAccumMacSkipped;
    return false;
  }

  int Src2Idx = AMDGPU::getNamedOperandIdx(MacOpc, AMDGPU::OpName::src2);
  assert(Src2Idx != -1 && "mac MFMA must have src2 operand");

  MI.setDesc(TII.get(MacOpc));
  Dest.setReg(Acc);
  Dest.setIsEarlyClobber(false);
  Src2->setReg(Acc);
  MI.tieOperands(0, Src2Idx);

  if (OldDest != Acc) {
    bool HasExternalUse = false;
    for (MachineOperand &MO : MRI.use_nodbg_operands(OldDest)) {
      if (MO.getParent() != &MI) {
        HasExternalUse = true;
        break;
      }
    }
    if (HasExternalUse) {
      MRI.replaceRegWith(OldDest, Acc);
      ++NumMFMAAccumMacDestRegsFolded;
    }
  }

  ++NumMFMAAccumMacConverted;
  return true;
}

class GCNMFMAAccumMacChainLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNMFMAAccumMacChainLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "GCN MFMA Accumulator Mac Chain";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

bool GCNMFMAAccumMacChainLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()) || !EnableMFMAAccumMacChain)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasMAIInsts())
    return false;

  const SIInstrInfo &TII = *ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (convertMFMAToMac(TII, MRI, MI))
        Changed = true;
    }
  }

  return Changed;
}

INITIALIZE_PASS_BEGIN(GCNMFMAAccumMacChainLegacy, DEBUG_TYPE,
                      "GCN MFMA Accumulator Mac Chain", false, false)
INITIALIZE_PASS_END(GCNMFMAAccumMacChainLegacy, DEBUG_TYPE,
                    "GCN MFMA Accumulator Mac Chain", false, false)

char GCNMFMAAccumMacChainLegacy::ID = 0;

char &llvm::GCNMFMAAccumMacChainID = GCNMFMAAccumMacChainLegacy::ID;

FunctionPass *llvm::createGCNMFMAAccumMacChainLegacyPass() {
  return new GCNMFMAAccumMacChainLegacy();
}

PreservedAnalyses
GCNMFMAAccumMacChainPass::run(MachineFunction &MF,
                              MachineFunctionAnalysisManager &MFAM) {
  if (!EnableMFMAAccumMacChain)
    return PreservedAnalyses::all();

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.hasMAIInsts())
    return PreservedAnalyses::all();

  const SIInstrInfo &TII = *ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (convertMFMAToMac(TII, MRI, MI))
        Changed = true;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
