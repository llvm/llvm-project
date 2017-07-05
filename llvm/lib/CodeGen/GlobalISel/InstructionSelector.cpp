//===- llvm/CodeGen/GlobalISel/InstructionSelector.cpp --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the InstructionSelector class.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/IR/Constants.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include <cassert>

#define DEBUG_TYPE "instructionselector"

using namespace llvm;

InstructionSelector::MatcherState::MatcherState(unsigned MaxRenderers)
    : Renderers(MaxRenderers, nullptr), MIs() {}

InstructionSelector::InstructionSelector() = default;

void InstructionSelector::executeEmitTable(NewMIVector &OutMIs,
                                           MatcherState &State,
                                           const int64_t *EmitTable,
                                           const TargetInstrInfo &TII,
                                           const TargetRegisterInfo &TRI,
                                           const RegisterBankInfo &RBI) const {
  const int64_t *Command = EmitTable;
  while (true) {
    switch (*Command++) {
    case GIR_MutateOpcode: {
      int64_t OldInsnID = *Command++;
      int64_t NewInsnID = *Command++;
      int64_t NewOpcode = *Command++;
      assert((size_t)NewInsnID == OutMIs.size() &&
             "Expected to store MIs in order");
      OutMIs.push_back(
          MachineInstrBuilder(*State.MIs[OldInsnID]->getParent()->getParent(),
                              State.MIs[OldInsnID]));
      OutMIs[NewInsnID]->setDesc(TII.get(NewOpcode));
      DEBUG(dbgs() << "GIR_MutateOpcode(OutMIs[" << NewInsnID << "], MIs["
                   << OldInsnID << "], " << NewOpcode << ")\n");
      break;
    }
    case GIR_BuildMI: {
      int64_t InsnID LLVM_ATTRIBUTE_UNUSED = *Command++;
      int64_t Opcode = *Command++;
      assert((size_t)InsnID == OutMIs.size() &&
             "Expected to store MIs in order");
      OutMIs.push_back(BuildMI(*State.MIs[0]->getParent(), State.MIs[0],
                               State.MIs[0]->getDebugLoc(), TII.get(Opcode)));
      DEBUG(dbgs() << "GIR_BuildMI(OutMIs[" << InsnID << "], " << Opcode
                   << ")\n");
      break;
    }

    case GIR_Copy: {
      int64_t NewInsnID = *Command++;
      int64_t OldInsnID = *Command++;
      int64_t OpIdx = *Command++;
      assert(OutMIs[NewInsnID] && "Attempted to add to undefined instruction");
      OutMIs[NewInsnID].add(State.MIs[OldInsnID]->getOperand(OpIdx));
      DEBUG(dbgs() << "GIR_Copy(OutMIs[" << NewInsnID << "], MIs[" << OldInsnID
                   << "], " << OpIdx << ")\n");
      break;
    }
    case GIR_CopySubReg: {
      int64_t NewInsnID = *Command++;
      int64_t OldInsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t SubRegIdx = *Command++;
      assert(OutMIs[NewInsnID] && "Attempted to add to undefined instruction");
      OutMIs[NewInsnID].addReg(State.MIs[OldInsnID]->getOperand(OpIdx).getReg(),
                               0, SubRegIdx);
      DEBUG(dbgs() << "GIR_CopySubReg(OutMIs[" << NewInsnID << "], MIs["
                   << OldInsnID << "], " << OpIdx << ", " << SubRegIdx
                   << ")\n");
      break;
    }
    case GIR_AddImplicitDef: {
      int64_t InsnID = *Command++;
      int64_t RegNum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addDef(RegNum, RegState::Implicit);
      DEBUG(dbgs() << "GIR_AddImplicitDef(OutMIs[" << InsnID << "], " << RegNum
                   << ")\n");
      break;
    }
    case GIR_AddImplicitUse: {
      int64_t InsnID = *Command++;
      int64_t RegNum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addUse(RegNum, RegState::Implicit);
      DEBUG(dbgs() << "GIR_AddImplicitUse(OutMIs[" << InsnID << "], " << RegNum
                   << ")\n");
      break;
    }
    case GIR_AddRegister: {
      int64_t InsnID = *Command++;
      int64_t RegNum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addReg(RegNum);
      DEBUG(dbgs() << "GIR_AddRegister(OutMIs[" << InsnID << "], " << RegNum
                   << ")\n");
      break;
    }
    case GIR_AddImm: {
      int64_t InsnID = *Command++;
      int64_t Imm = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      OutMIs[InsnID].addImm(Imm);
      DEBUG(dbgs() << "GIR_AddImm(OutMIs[" << InsnID << "], " << Imm << ")\n");
      break;
    }
    case GIR_ComplexRenderer: {
      int64_t InsnID = *Command++;
      int64_t RendererID = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      State.Renderers[RendererID](OutMIs[InsnID]);
      DEBUG(dbgs() << "GIR_ComplexRenderer(OutMIs[" << InsnID << "], "
                   << RendererID << ")\n");
      break;
    }

    case GIR_ConstrainOperandRC: {
      int64_t InsnID = *Command++;
      int64_t OpIdx = *Command++;
      int64_t RCEnum = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      constrainOperandRegToRegClass(*OutMIs[InsnID].getInstr(), OpIdx,
                                    *TRI.getRegClass(RCEnum), TII, TRI, RBI);
      DEBUG(dbgs() << "GIR_ConstrainOperandRC(OutMIs[" << InsnID << "], "
                   << OpIdx << ", " << RCEnum << ")\n");
      break;
    }
    case GIR_ConstrainSelectedInstOperands: {
      int64_t InsnID = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      constrainSelectedInstRegOperands(*OutMIs[InsnID].getInstr(), TII, TRI,
                                       RBI);
      DEBUG(dbgs() << "GIR_ConstrainSelectedInstOperands(OutMIs[" << InsnID
                   << "])\n");
      break;
    }
    case GIR_MergeMemOperands: {
      int64_t InsnID = *Command++;
      assert(OutMIs[InsnID] && "Attempted to add to undefined instruction");
      for (const auto *FromMI : State.MIs)
        for (const auto &MMO : FromMI->memoperands())
          OutMIs[InsnID].addMemOperand(MMO);
      DEBUG(dbgs() << "GIR_MergeMemOperands(OutMIs[" << InsnID << "])\n");
      break;
    }
    case GIR_EraseFromParent: {
      int64_t InsnID = *Command++;
      assert(State.MIs[InsnID] && "Attempted to erase an undefined instruction");
      State.MIs[InsnID]->eraseFromParent();
      DEBUG(dbgs() << "GIR_EraseFromParent(MIs[" << InsnID << "])\n");
      break;
    }

    case GIR_Done:
      DEBUG(dbgs() << "GIR_Done");
      return;
    default:
      llvm_unreachable("Unexpected command");
    }
  }
}

bool InstructionSelector::constrainOperandRegToRegClass(
    MachineInstr &I, unsigned OpIdx, const TargetRegisterClass &RC,
    const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
    const RegisterBankInfo &RBI) const {
  MachineBasicBlock &MBB = *I.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  return
      constrainRegToClass(MRI, TII, RBI, I, I.getOperand(OpIdx).getReg(), RC);
}

bool InstructionSelector::constrainSelectedInstRegOperands(
    MachineInstr &I, const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
    const RegisterBankInfo &RBI) const {
  MachineBasicBlock &MBB = *I.getParent();
  MachineFunction &MF = *MBB.getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  for (unsigned OpI = 0, OpE = I.getNumExplicitOperands(); OpI != OpE; ++OpI) {
    MachineOperand &MO = I.getOperand(OpI);

    // There's nothing to be done on non-register operands.
    if (!MO.isReg())
      continue;

    DEBUG(dbgs() << "Converting operand: " << MO << '\n');
    assert(MO.isReg() && "Unsupported non-reg operand");

    unsigned Reg = MO.getReg();
    // Physical registers don't need to be constrained.
    if (TRI.isPhysicalRegister(Reg))
      continue;

    // Register operands with a value of 0 (e.g. predicate operands) don't need
    // to be constrained.
    if (Reg == 0)
      continue;

    // If the operand is a vreg, we should constrain its regclass, and only
    // insert COPYs if that's impossible.
    // constrainOperandRegClass does that for us.
    MO.setReg(constrainOperandRegClass(MF, TRI, MRI, TII, RBI, I, I.getDesc(),
                                       Reg, OpI));

    // Tie uses to defs as indicated in MCInstrDesc if this hasn't already been
    // done.
    if (MO.isUse()) {
      int DefIdx = I.getDesc().getOperandConstraint(OpI, MCOI::TIED_TO);
      if (DefIdx != -1 && !I.isRegTiedToUseOperand(DefIdx))
        I.tieOperands(DefIdx, OpI);
    }
  }
  return true;
}

bool InstructionSelector::isOperandImmEqual(
    const MachineOperand &MO, int64_t Value,
    const MachineRegisterInfo &MRI) const {
  if (MO.isReg() && MO.getReg())
    if (auto VRegVal = getConstantVRegVal(MO.getReg(), MRI))
      return *VRegVal == Value;
  return false;
}

bool InstructionSelector::isObviouslySafeToFold(MachineInstr &MI) const {
  return !MI.mayLoadOrStore() && !MI.hasUnmodeledSideEffects() &&
         MI.implicit_operands().begin() == MI.implicit_operands().end();
}
