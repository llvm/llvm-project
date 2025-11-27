//==-- ARMConvertToComparesPass.cpp - Convert dead dests to compares --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file When allowed by the instruction, replace dead definitions with compare
/// instructions.
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMBaseInstrInfo.h"
#include "ARMBaseRegisterInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
using namespace llvm;

#define DEBUG_TYPE "arm-convert-to-cmp"

STATISTIC(NumDeadDefsReplaced, "Number of instructions converted");

#define ARM_DEAD_REG_DEF_NAME "Convert Dead Defs to Compare Instructions"

namespace {
class ARMConvertToCompares : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  bool Changed;
  void processMachineBasicBlock(MachineBasicBlock &MBB);

public:
  static char ID; // Pass identification, replacement for typeid.
  ARMConvertToCompares() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override { return ARM_DEAD_REG_DEF_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
char ARMConvertToCompares::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(ARMConvertToCompares, "arm-dead-defs-to-cmp",
                ARM_DEAD_REG_DEF_NAME, false, false)

static bool usesFrameIndex(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.uses())
    if (MO.isFI())
      return true;
  return false;
}

static std::optional<unsigned> mapToCmpCmnTstTeqOpcode(unsigned Opc) {
  switch (Opc) {
  // ARM encodings
  case ARM::SUBri:
    return ARM::CMPri;
  case ARM::SUBrr:
    return ARM::CMPrr;
  case ARM::SUBrsi:
    return ARM::CMPrsi;
  case ARM::SUBrsr:
    return ARM::CMPrsr;

  case ARM::ADDri:
    return ARM::CMNri;
  case ARM::ADDrr:
    return ARM::CMNzrr;
  case ARM::ADDrsi:
    return ARM::CMNzrsi;
  case ARM::ADDrsr:
    return ARM::CMNzrsr;

  case ARM::ANDri:
    return ARM::TSTri;
  case ARM::ANDrr:
    return ARM::TSTrr;
  case ARM::ANDrsi:
    return ARM::TSTrsi;
  case ARM::ANDrsr:
    return ARM::TSTrsr;

  case ARM::EORri:
    return ARM::TEQri;
  case ARM::EORrr:
    return ARM::TEQrr;
  case ARM::EORrsi:
    return ARM::TEQrsi;
  case ARM::EORrsr:
    return ARM::TEQrsr;

  // Thumb2 encodings
  case ARM::t2SUBri:
    return ARM::t2CMPri;
  case ARM::t2SUBrr:
    return ARM::t2CMPrr;
  case ARM::t2SUBrs:
    return ARM::t2CMPrs;

  case ARM::t2ADDri:
    return ARM::t2CMNri;
  case ARM::t2ADDrr:
    return ARM::t2CMNzrr;
  case ARM::t2ADDrs:
    return ARM::t2CMNzrs;

  case ARM::t2ANDri:
    return ARM::t2TSTri;
  case ARM::t2ANDrr:
    return ARM::t2TSTrr;
  case ARM::t2ANDrs:
    return ARM::t2TSTrs;

  case ARM::t2EORri:
    return ARM::t2TEQri;
  case ARM::t2EORrr:
    return ARM::t2TEQrr;
  case ARM::t2EORrs:
    return ARM::t2TEQrs;

  // Thumb1 limited support
  case ARM::tSUBSrr:
    return ARM::tCMPr;

  // Source is unused anyway so both go to tCMPi8
  case ARM::tSUBSi3:
  case ARM::tSUBSi8:
    return ARM::tCMPi8;
  case ARM::tADDSrr:
    return ARM::tCMNz; // At this point, flags don't matter. tCMNz is CMN.
  case ARM::tAND:
    return ARM::tTST;
  default:
    return std::nullopt;
  }
}

static void copyNonDefNonPredOperands(MachineInstr &Dst,
                                      const MachineInstr &Src) {
  const MCInstrDesc &Desc = Src.getDesc();
  int PIdx = Src.findFirstPredOperandIdx();
  unsigned Start = Desc.getNumDefs();
  unsigned End =
      (PIdx == -1) ? Src.getNumOperands() : static_cast<unsigned>(PIdx);
  for (unsigned I = Start; I < End; ++I)
    Dst.addOperand(Src.getOperand(I));
  if (PIdx != -1) {
    Dst.addOperand(Src.getOperand(PIdx));
    Dst.addOperand(Src.getOperand(PIdx + 1));
  }
}

void ARMConvertToCompares::processMachineBasicBlock(MachineBasicBlock &MBB) {
  // Early-increment range: iterator is advanced before the loop body, so it's
  // safe to erase the current instruction inside the loop.
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    if (usesFrameIndex(MI))
      continue;

    // Only consider instructions that set CPSR (flag-setting variants).
    if (!ARMBaseInstrInfo::isCPSRDefined(MI))
      continue;

    const MCInstrDesc &Desc = MI.getDesc();

    for (int I = 0, EE = Desc.getNumDefs(); I != EE; ++I) {
      MachineOperand &MO = MI.getOperand(I);
      if (!MO.isReg() || !MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isVirtual() || (!MO.isDead() && !MRI->use_nodbg_empty(Reg)))
        continue;
      assert(!MO.isImplicit() && "Unexpected implicit def!");
      if (MI.isRegTiedToUseOperand(I))
        continue;

      if (std::optional<unsigned> NewOpc =
              mapToCmpCmnTstTeqOpcode(MI.getOpcode())) {
        MachineInstrBuilder MIB =
            BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(*NewOpc));
        copyNonDefNonPredOperands(*MIB, MI);
        MIB.setMIFlags(MI.getFlags());
        for (MachineMemOperand *MMO : MI.memoperands())
          MIB.addMemOperand(MMO);

        MI.eraseFromParent();
        ++NumDeadDefsReplaced;
        Changed = true;
        break;
      }
    }
  }
}

// Scan the function for instructions that have a dead definition of a
// register. Replace that instruction with a compare instruction when possible
bool ARMConvertToCompares::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  LLVM_DEBUG(dbgs() << "***** ARMConvertToComparesPass *****\n");
  Changed = false;
  for (auto &MBB : MF)
    processMachineBasicBlock(MBB);
  return Changed;
}

FunctionPass *llvm::createARMConvertToComparesPass() {
  return new ARMConvertToCompares();
}
