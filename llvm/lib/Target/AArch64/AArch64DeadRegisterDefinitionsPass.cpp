//==-- AArch64DeadRegisterDefinitions.cpp - Replace dead defs w/ zero reg --==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file When allowed by the instruction, replace a dead definition of a GPR
/// with the zero register. This makes the code a bit friendlier towards the
/// hardware's register renamer.
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64RegisterInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "aarch64-dead-defs"

STATISTIC(NumDeadDefsReplaced, "Number of dead definitions replaced");

#define AARCH64_DEAD_REG_DEF_NAME "AArch64 Dead register definitions"

namespace {
class AArch64DeadRegisterDefinitions : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  bool Changed;
  void processMachineBasicBlock(MachineBasicBlock &MBB);
public:
  static char ID; // Pass identification, replacement for typeid.
  AArch64DeadRegisterDefinitions() : MachineFunctionPass(ID) {
    initializeAArch64DeadRegisterDefinitionsPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  StringRef getPassName() const override { return AARCH64_DEAD_REG_DEF_NAME; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
char AArch64DeadRegisterDefinitions::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(AArch64DeadRegisterDefinitions, "aarch64-dead-defs",
                AARCH64_DEAD_REG_DEF_NAME, false, false)

static bool usesFrameIndex(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.uses())
    if (MO.isFI())
      return true;
  return false;
}

void AArch64DeadRegisterDefinitions::processMachineBasicBlock(
    MachineBasicBlock &MBB) {
  const MachineFunction &MF = *MBB.getParent();
  for (MachineInstr &MI : MBB) {
    if (usesFrameIndex(MI)) {
      // We need to skip this instruction because while it appears to have a
      // dead def it uses a frame index which might expand into a multi
      // instruction sequence during EPI.
      DEBUG(dbgs() << "    Ignoring, operand is frame index\n");
      continue;
    }
    if (MI.definesRegister(AArch64::XZR) || MI.definesRegister(AArch64::WZR)) {
      // It is not allowed to write to the same register (not even the zero
      // register) twice in a single instruction.
      DEBUG(dbgs() << "    Ignoring, XZR or WZR already used by the instruction\n");
      continue;
    }
    if (MF.getSubtarget<AArch64Subtarget>().hasLSE()) {
      // XZ/WZ for LSE can only be used when acquire semantics are not used,
      // LDOPAL WZ is an invalid opcode.
      switch (MI.getOpcode()) {
      case AArch64::CASALB:
      case AArch64::CASALH:
      case AArch64::CASALW:
      case AArch64::CASALX:
      case AArch64::SWPALB:
      case AArch64::SWPALH:
      case AArch64::SWPALW:
      case AArch64::SWPALX:
      case AArch64::LDADDALB:
      case AArch64::LDADDALH:
      case AArch64::LDADDALW:
      case AArch64::LDADDALX:
      case AArch64::LDCLRALB:
      case AArch64::LDCLRALH:
      case AArch64::LDCLRALW:
      case AArch64::LDCLRALX:
      case AArch64::LDEORALB:
      case AArch64::LDEORALH:
      case AArch64::LDEORALW:
      case AArch64::LDEORALX:
      case AArch64::LDSETALB:
      case AArch64::LDSETALH:
      case AArch64::LDSETALW:
      case AArch64::LDSETALX:
      case AArch64::LDSMINALB:
      case AArch64::LDSMINALH:
      case AArch64::LDSMINALW:
      case AArch64::LDSMINALX:
      case AArch64::LDSMAXALB:
      case AArch64::LDSMAXALH:
      case AArch64::LDSMAXALW:
      case AArch64::LDSMAXALX:
      case AArch64::LDUMINALB:
      case AArch64::LDUMINALH:
      case AArch64::LDUMINALW:
      case AArch64::LDUMINALX:
      case AArch64::LDUMAXALB:
      case AArch64::LDUMAXALH:
      case AArch64::LDUMAXALW:
      case AArch64::LDUMAXALX:
        continue;
      default:
        break;
      }
    }
    const MCInstrDesc &Desc = MI.getDesc();
    for (int I = 0, E = Desc.getNumDefs(); I != E; ++I) {
      MachineOperand &MO = MI.getOperand(I);
      if (!MO.isReg() || !MO.isDef())
        continue;
      // We should not have any relevant physreg defs that are replacable by
      // zero before register allocation. So we just check for dead vreg defs.
      unsigned Reg = MO.getReg();
      if (!TargetRegisterInfo::isVirtualRegister(Reg) ||
          (!MO.isDead() && !MRI->use_nodbg_empty(Reg)))
        continue;
      assert(!MO.isImplicit() && "Unexpected implicit def!");
      DEBUG(dbgs() << "  Dead def operand #" << I << " in:\n    ";
            MI.print(dbgs()));
      // Be careful not to change the register if it's a tied operand.
      if (MI.isRegTiedToUseOperand(I)) {
        DEBUG(dbgs() << "    Ignoring, def is tied operand.\n");
        continue;
      }
      const TargetRegisterClass *RC = TII->getRegClass(Desc, I, TRI, MF);
      unsigned NewReg;
      if (RC == nullptr) {
        DEBUG(dbgs() << "    Ignoring, register is not a GPR.\n");
        continue;
      } else if (RC->contains(AArch64::WZR))
        NewReg = AArch64::WZR;
      else if (RC->contains(AArch64::XZR))
        NewReg = AArch64::XZR;
      else {
        DEBUG(dbgs() << "    Ignoring, register is not a GPR.\n");
        continue;
      }
      DEBUG(dbgs() << "    Replacing with zero register. New:\n      ");
      MO.setReg(NewReg);
      MO.setIsDead();
      DEBUG(MI.print(dbgs()));
      ++NumDeadDefsReplaced;
      Changed = true;
      // Only replace one dead register, see check for zero register above.
      break;
    }
  }
}

// Scan the function for instructions that have a dead definition of a
// register. Replace that register with the zero register when possible.
bool AArch64DeadRegisterDefinitions::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  DEBUG(dbgs() << "***** AArch64DeadRegisterDefinitions *****\n");
  Changed = false;
  for (auto &MBB : MF)
    processMachineBasicBlock(MBB);
  return Changed;
}

FunctionPass *llvm::createAArch64DeadRegisterDefinitions() {
  return new AArch64DeadRegisterDefinitions();
}
