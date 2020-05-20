//===-- MVEVPTOptimisationsPass.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass does a few optimisations related to MVE VPT blocks before
/// register allocation is performed. The goal is to maximize the sizes of the
/// blocks that will be created by the MVE VPT Block Insertion pass (which runs
/// after register allocation). Currently, this pass replaces VCMPs with VPNOTs
/// when possible, so the Block Insertion pass can delete them later to create
/// larger VPT blocks.
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMSubtarget.h"
#include "MCTargetDesc/ARMBaseInfo.h"
#include "Thumb2InstrInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/Debug.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "arm-mve-vpt-opts"

namespace {
class MVEVPTOptimisations : public MachineFunctionPass {
public:
  static char ID;
  const Thumb2InstrInfo *TII;
  MachineRegisterInfo *MRI;

  MVEVPTOptimisations() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &Fn) override;

  StringRef getPassName() const override {
    return "ARM MVE VPT Optimisation Pass";
  }

private:
  bool ReplaceVCMPsByVPNOTs(MachineBasicBlock &MBB);
};

char MVEVPTOptimisations::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(MVEVPTOptimisations, DEBUG_TYPE,
                "ARM MVE VPT Optimisations pass", false, false)

// Returns true if Opcode is any VCMP Opcode.
static bool IsVCMP(unsigned Opcode) { return VCMPOpcodeToVPT(Opcode) != 0; }

// Returns true if a VCMP with this Opcode can have its operands swapped.
// There is 2 kind of VCMP that can't have their operands swapped: Float VCMPs,
// and VCMPr instructions (since the r is always on the right).
static bool CanHaveSwappedOperands(unsigned Opcode) {
  switch (Opcode) {
  default:
    return true;
  case ARM::MVE_VCMPf32:
  case ARM::MVE_VCMPf16:
  case ARM::MVE_VCMPf32r:
  case ARM::MVE_VCMPf16r:
  case ARM::MVE_VCMPi8r:
  case ARM::MVE_VCMPi16r:
  case ARM::MVE_VCMPi32r:
  case ARM::MVE_VCMPu8r:
  case ARM::MVE_VCMPu16r:
  case ARM::MVE_VCMPu32r:
  case ARM::MVE_VCMPs8r:
  case ARM::MVE_VCMPs16r:
  case ARM::MVE_VCMPs32r:
    return false;
  }
}

// Returns the CondCode of a VCMP Instruction.
static ARMCC::CondCodes GetCondCode(MachineInstr &Instr) {
  assert(IsVCMP(Instr.getOpcode()) && "Inst must be a VCMP");
  return ARMCC::CondCodes(Instr.getOperand(3).getImm());
}

// Returns true if Cond is equivalent to a VPNOT instruction on the result of
// Prev. Cond and Prev must be VCMPs.
static bool IsVPNOTEquivalent(MachineInstr &Cond, MachineInstr &Prev) {
  assert(IsVCMP(Cond.getOpcode()) && IsVCMP(Prev.getOpcode()));

  // Opcodes must match.
  if (Cond.getOpcode() != Prev.getOpcode())
    return false;

  MachineOperand &CondOP1 = Cond.getOperand(1), &CondOP2 = Cond.getOperand(2);
  MachineOperand &PrevOP1 = Prev.getOperand(1), &PrevOP2 = Prev.getOperand(2);

  // If the VCMP has the opposite condition with the same operands, we can
  // replace it with a VPNOT
  ARMCC::CondCodes ExpectedCode = GetCondCode(Cond);
  ExpectedCode = ARMCC::getOppositeCondition(ExpectedCode);
  if (ExpectedCode == GetCondCode(Prev))
    if (CondOP1.isIdenticalTo(PrevOP1) && CondOP2.isIdenticalTo(PrevOP2))
      return true;
  // Check again with operands swapped if possible
  if (!CanHaveSwappedOperands(Cond.getOpcode()))
    return false;
  ExpectedCode = ARMCC::getSwappedCondition(ExpectedCode);
  return ExpectedCode == GetCondCode(Prev) && CondOP1.isIdenticalTo(PrevOP2) &&
         CondOP2.isIdenticalTo(PrevOP1);
}

// Returns true if Instr writes to VCCR.
static bool IsWritingToVCCR(MachineInstr &Instr) {
  if (Instr.getNumOperands() == 0)
    return false;
  MachineOperand &Dst = Instr.getOperand(0);
  if (!Dst.isReg())
    return false;
  Register DstReg = Dst.getReg();
  if (!DstReg.isVirtual())
    return false;
  MachineRegisterInfo &RegInfo = Instr.getMF()->getRegInfo();
  const TargetRegisterClass *RegClass = RegInfo.getRegClassOrNull(DstReg);
  return RegClass && (RegClass->getID() == ARM::VCCRRegClassID);
}

// This optimisation replaces VCMPs with VPNOTs when they are equivalent.
bool MVEVPTOptimisations::ReplaceVCMPsByVPNOTs(MachineBasicBlock &MBB) {
  SmallVector<MachineInstr *, 4> DeadInstructions;

  // The last VCMP that we have seen and that couldn't be replaced.
  // This is reset when an instruction that writes to VCCR/VPR is found, or when
  // a VCMP is replaced with a VPNOT.
  // We'll only replace VCMPs with VPNOTs when this is not null, and when the
  // current VCMP is the opposite of PrevVCMP.
  MachineInstr *PrevVCMP = nullptr;
  // If we find an instruction that kills the result of PrevVCMP, we save the
  // operand here to remove the kill flag in case we need to use PrevVCMP's
  // result.
  MachineOperand *PrevVCMPResultKiller = nullptr;

  for (MachineInstr &Instr : MBB.instrs()) {
    if (PrevVCMP) {
      if (MachineOperand *MO = Instr.findRegisterUseOperand(
              PrevVCMP->getOperand(0).getReg(), /*isKill*/ true)) {
        // If we come accross the instr that kills PrevVCMP's result, record it
        // so we can remove the kill flag later if we need to.
        PrevVCMPResultKiller = MO;
      }
    }

    // Ignore predicated instructions.
    if (getVPTInstrPredicate(Instr) != ARMVCC::None)
      continue;

    // Only look at VCMPs
    if (!IsVCMP(Instr.getOpcode())) {
      // If the instruction writes to VCCR, forget the previous VCMP.
      if (IsWritingToVCCR(Instr))
        PrevVCMP = nullptr;
      continue;
    }

    if (!PrevVCMP || !IsVPNOTEquivalent(Instr, *PrevVCMP)) {
      PrevVCMP = &Instr;
      continue;
    }

    // The register containing the result of the VCMP that we're going to
    // replace.
    Register PrevVCMPResultReg = PrevVCMP->getOperand(0).getReg();

    // Build a VPNOT to replace the VCMP, reusing its operands.
    MachineInstrBuilder MIBuilder =
        BuildMI(MBB, &Instr, Instr.getDebugLoc(), TII->get(ARM::MVE_VPNOT))
            .add(Instr.getOperand(0))
            .addReg(PrevVCMPResultReg);
    addUnpredicatedMveVpredNOp(MIBuilder);
    LLVM_DEBUG(dbgs() << "Inserting VPNOT (to replace VCMP): ";
               MIBuilder.getInstr()->dump(); dbgs() << "  Removed VCMP: ";
               Instr.dump());

    // If we found an instruction that uses, and kills PrevVCMP's result,
    // remove the kill flag.
    if (PrevVCMPResultKiller)
      PrevVCMPResultKiller->setIsKill(false);

    // Finally, mark the old VCMP for removal and reset
    // PrevVCMP/PrevVCMPResultKiller.
    DeadInstructions.push_back(&Instr);
    PrevVCMP = nullptr;
    PrevVCMPResultKiller = nullptr;
  }

  for (MachineInstr *DeadInstruction : DeadInstructions)
    DeadInstruction->removeFromParent();

  return !DeadInstructions.empty();
}

bool MVEVPTOptimisations::runOnMachineFunction(MachineFunction &Fn) {
  const ARMSubtarget &STI =
      static_cast<const ARMSubtarget &>(Fn.getSubtarget());

  if (!STI.isThumb2() || !STI.hasMVEIntegerOps())
    return false;

  TII = static_cast<const Thumb2InstrInfo *>(STI.getInstrInfo());
  MRI = &Fn.getRegInfo();

  LLVM_DEBUG(dbgs() << "********** ARM MVE VPT Optimisations **********\n"
                    << "********** Function: " << Fn.getName() << '\n');

  bool Modified = false;
  for (MachineBasicBlock &MBB : Fn)
    Modified |= ReplaceVCMPsByVPNOTs(MBB);

  LLVM_DEBUG(dbgs() << "**************************************\n");
  return Modified;
}

/// createMVEVPTOptimisationsPass
FunctionPass *llvm::createMVEVPTOptimisationsPass() {
  return new MVEVPTOptimisations();
}
