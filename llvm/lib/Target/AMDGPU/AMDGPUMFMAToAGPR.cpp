//===-- AMDGPUMFMAToAGPR.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMFMAToAGPR.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mfma-to-agpr"

bool AMDGPU::hasAGPRForm(const MachineInstr &MI, const SIInstrInfo &TII) {
  return TII.isMAI(MI) && AMDGPU::getAGPRFormOp(MI.getOpcode()) != -1;
}

/// getRegClassConstraintEffect() against a descriptor the instruction does not
/// carry yet the AGPR-form one.
static const TargetRegisterClass *
constrainByDesc(const TargetRegisterClass *CurRC, const MCInstrDesc &Desc,
                const MachineOperand &MO, const SIInstrInfo &TII,
                const SIRegisterInfo &TRI) {
  const TargetRegisterClass *OpRC = TII.getRegClass(Desc, MO.getOperandNo());
  if (!OpRC)
    return CurRC;
  if (unsigned SubIdx = MO.getSubReg())
    return TRI.getMatchingSuperRegClass(CurRC, OpRC, SubIdx);
  return TRI.getCommonSubClass(CurRC, OpRC);
}

bool AMDGPU::MFMAToAGPR::isActive(const MachineInstr &MI) const {
  if (!hasAGPRForm(MI, TII))
    return false;
  return !ActiveSet || ActiveSet->contains(&MI);
}

bool AMDGPU::MFMAToAGPR::operandAcceptsClass(
    const MachineOperand &MO, const TargetRegisterClass *RC) const {
  const MachineInstr &MI = *MO.getParent();
  if (isActive(MI)) {
    const MCInstrDesc &AGPRDesc =
        TII.get(AMDGPU::getAGPRFormOp(MI.getOpcode()));
    return constrainByDesc(RC, AGPRDesc, MO, TII, TRI);
  }
  return MI.getRegClassConstraintEffect(MO.getOperandNo(), RC, &TII, &TRI);
}

bool AMDGPU::MFMAToAGPR::canRecolorToAGPR(
    Register Reg, ArrayRef<MachineOperand *> ExemptOps,
    SmallVectorImpl<MachineInstr *> *Chain,
    SmallSetVector<Register, 4> *ChainRegs) const {
  if (Reg.isPhysical())
    return false;

  // Both callers install the equivalent AGPR class, so there has to be one.
  const TargetRegisterClass *OldRC = MRI.getRegClass(Reg);
  if (!TRI.getEquivalentAGPRClass(OldRC))
    return false;

  // Start from the widest class the subtarget allows and let the operands
  // narrow it. Staying at OldRC means nothing admitted an AGPR.
  const TargetRegisterClass *NewRC = TRI.getLargestLegalSuperClass(OldRC, MF);
  if (!NewRC || OldRC == NewRC)
    return false;

  // Accumulate constraints from all defs and uses.
  for (MachineOperand &MO : MRI.reg_nodbg_operands(Reg)) {
    MachineInstr *MI = MO.getParent();

    // An MFMA we are converting constrains Reg by its AGPR-form operand class,
    // not the VGPR-form one it still carries. This lets dst and src2 swap as a
    // pair while the scale variant's VGPR-only src0/src1 stay put.
    if (isActive(*MI)) {
      const MCInstrDesc &AGPRDesc =
          TII.get(AMDGPU::getAGPRFormOp(MI->getOpcode()));
      NewRC = constrainByDesc(NewRC, AGPRDesc, MO, TII, TRI);
      if (!NewRC || NewRC == OldRC) {
        LLVM_DEBUG(dbgs() << "User of " << printReg(Reg, &TRI)
                          << " has no AGPR form accepting it: " << *MI);
        return false;
      }

      const MachineOperand *VDst =
          TII.getNamedOperand(*MI, AMDGPU::OpName::vdst);
      const MachineOperand *Src2 =
          TII.getNamedOperand(*MI, AMDGPU::OpName::src2);
      for (const MachineOperand *Op : {VDst, Src2}) {
        if (!Op->isReg())
          continue;
        if (Op->getReg().isPhysical())
          return false;
        if (ChainRegs && Op->getReg() != Reg)
          ChainRegs->insert(Op->getReg());
      }

      if (Chain && !is_contained(*Chain, MI))
        Chain->push_back(MI);

      continue;
    }

    // An MFMA left in VGPR form pins Reg to the VGPR bank. Its operand class
    // nominally admits an AGPR, so the check below would wrongly accept it.
    if (hasAGPRForm(*MI, TII)) {
      LLVM_DEBUG(dbgs() << "User of " << printReg(Reg, &TRI)
                        << " is an MFMA left in VGPR form: " << *MI);
      return false;
    }

    // Uses the caller will redirect to a bridge copy do not constrain Reg.
    if (is_contained(ExemptOps, &MO))
      continue;

    NewRC =
        MI->getRegClassConstraintEffect(MO.getOperandNo(), NewRC, &TII, &TRI);
    if (!NewRC || NewRC == OldRC) {
      LLVM_DEBUG(dbgs() << "User of " << printReg(Reg, &TRI)
                        << " cannot be reassigned to "
                        << (NewRC ? TRI.getRegClassName(NewRC) : "NULL") << ": "
                        << *MI);
      return false;
    }
  }

  return true;
}

bool AMDGPU::MFMAToAGPR::canMoveChainToAGPR(
    Register StartReg, SmallVectorImpl<MachineInstr *> &Chain,
    SmallSetVector<Register, 4> &ChainRegs,
    ArrayRef<MachineOperand *> ExemptOps) const {
  SmallVector<Register, 8> Worklist = {StartReg};

  // Recursively visit all transitive MFMA users.
  while (!Worklist.empty()) {
    Register Reg = Worklist.pop_back_val();
    unsigned NumChainRegs = ChainRegs.size();

    if (!canRecolorToAGPR(Reg, ExemptOps, &Chain, &ChainRegs))
      return false;

    // Registers canRecolorToAGPR just discovered still need visiting.
    for (Register NewReg : ChainRegs.getArrayRef().drop_front(NumChainRegs))
      Worklist.push_back(NewReg);
  }

  return true;
}

SmallVector<SmallVector<MachineInstr *, 8>, 16>
AMDGPU::MFMAToAGPR::partitionChains() const {
  SmallVector<MachineInstr *, 32> Candidates;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (hasAGPRForm(MI, TII))
        Candidates.push_back(&MI);

  // Join MFMAs linked by a dst -> src2 edge.
  EquivalenceClasses<MachineInstr *> Classes;
  for (MachineInstr *MI : Candidates)
    Classes.insert(MI);

  for (MachineInstr *MI : Candidates) {
    const MachineOperand *Src2 = TII.getNamedOperand(*MI, AMDGPU::OpName::src2);
    if (!Src2 || !Src2->isReg() || !Src2->getReg().isVirtual())
      continue;
    for (MachineInstr &Def : MRI.def_instructions(Src2->getReg()))
      if (hasAGPRForm(Def, TII))
        Classes.unionSets(&Def, MI);
  }

  // Candidates is in program order, so both the chains and their members come
  // out in program order.
  MapVector<MachineInstr *, SmallVector<MachineInstr *, 8>> Grouped;
  for (MachineInstr *MI : Candidates)
    Grouped[Classes.getLeaderValue(MI)].push_back(MI);

  SmallVector<SmallVector<MachineInstr *, 8>, 16> Chains;
  Chains.reserve(Grouped.size());
  for (auto &KV : Grouped)
    Chains.push_back(std::move(KV.second));
  return Chains;
}

void AMDGPU::MFMAToAGPR::applyAGPRForm(
    ArrayRef<MachineInstr *> MFMAs, ArrayRef<Register> Regs,
    SmallVectorImpl<unsigned> *OrigOpcodes) const {
  for (Register Reg : Regs) {
    const TargetRegisterClass *AGPRRC =
        TRI.getEquivalentAGPRClass(MRI.getRegClass(Reg));
    assert(AGPRRC && "register has no AGPR equivalent class");
    MRI.setRegClass(Reg, AGPRRC);
  }

  for (MachineInstr *MI : MFMAs) {
    int NewOp = AMDGPU::getAGPRFormOp(MI->getOpcode());
    assert(NewOp != -1 && "MFMA has no AGPR form");
    if (OrigOpcodes)
      OrigOpcodes->push_back(MI->getOpcode());
    MI->setDesc(TII.get(NewOp));
  }
}

void AMDGPU::MFMAToAGPR::revertToVGPRForm(ArrayRef<MachineInstr *> MFMAs,
                                          ArrayRef<unsigned> OrigOpcodes,
                                          ArrayRef<Register> Regs) const {
  assert(MFMAs.size() == OrigOpcodes.size() &&
         "opcode list does not match the instruction list");
  for (auto [MI, Opcode] : zip_equal(MFMAs, OrigOpcodes))
    MI->setDesc(TII.get(Opcode));

  for (Register Reg : Regs) {
    const TargetRegisterClass *VGPRRC =
        TRI.getEquivalentVGPRClass(MRI.getRegClass(Reg));
    assert(VGPRRC && "register has no VGPR equivalent class");
    MRI.setRegClass(Reg, VGPRRC);
  }
}

AMDGPU::MFMAToAGPR::FormScope::FormScope(const MFMAToAGPR &Impl,
                                         ArrayRef<MachineInstr *> MFMAs,
                                         ArrayRef<Register> Regs)
    : Impl(Impl), MFMAs(MFMAs), Regs(Regs) {
  OrigOpcodes.reserve(MFMAs.size());
  Impl.applyAGPRForm(MFMAs, Regs, &OrigOpcodes);
}

AMDGPU::MFMAToAGPR::FormScope::~FormScope() {
  if (Active)
    Impl.revertToVGPRForm(MFMAs, OrigOpcodes, Regs);
}
