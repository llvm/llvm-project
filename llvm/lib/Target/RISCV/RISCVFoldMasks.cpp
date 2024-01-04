//===- RISCVFoldMasks.cpp - MI Vector Pseudo Mask Peepholes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass performs various peephole optimisations that fold masks into vector
// pseudo instructions after instruction selection.
//
// Currently it converts
// PseudoVMERGE_VVM %false, %false, %true, %allonesmask, %vl, %sew
// ->
// PseudoVMV_V_V %false, %true, %vl, %sew
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVISelDAGToDAG.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-fold-masks"

namespace {

class RISCVFoldMasks : public MachineFunctionPass {
public:
  static char ID;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  RISCVFoldMasks() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  StringRef getPassName() const override { return "RISC-V Fold Masks"; }

private:
  bool convertToUnmasked(MachineInstr &MI, MachineInstr *MaskDef);
  bool convertVMergeToVMv(MachineInstr &MI, MachineInstr *MaskDef);

  bool isAllOnesMask(MachineInstr *MaskDef);
};

} // namespace

char RISCVFoldMasks::ID = 0;

INITIALIZE_PASS(RISCVFoldMasks, DEBUG_TYPE, "RISC-V Fold Masks", false, false)

bool RISCVFoldMasks::isAllOnesMask(MachineInstr *MaskDef) {
  if (!MaskDef)
    return false;
  assert(MaskDef->isCopy() && MaskDef->getOperand(0).getReg() == RISCV::V0);
  Register SrcReg = TRI->lookThruCopyLike(MaskDef->getOperand(1).getReg(), MRI);
  if (!SrcReg.isVirtual())
    return false;
  MaskDef = MRI->getVRegDef(SrcReg);
  if (!MaskDef)
    return false;

  // TODO: Check that the VMSET is the expected bitwidth? The pseudo has
  // undefined behaviour if it's the wrong bitwidth, so we could choose to
  // assume that it's all-ones? Same applies to its VL.
  switch (MaskDef->getOpcode()) {
  case RISCV::PseudoVMSET_M_B1:
  case RISCV::PseudoVMSET_M_B2:
  case RISCV::PseudoVMSET_M_B4:
  case RISCV::PseudoVMSET_M_B8:
  case RISCV::PseudoVMSET_M_B16:
  case RISCV::PseudoVMSET_M_B32:
  case RISCV::PseudoVMSET_M_B64:
    return true;
  default:
    return false;
  }
}

// Transform (VMERGE_VVM_<LMUL> false, false, true, allones, vl, sew) to
// (VMV_V_V_<LMUL> false, true, vl, sew). It may decrease uses of VMSET.
bool RISCVFoldMasks::convertVMergeToVMv(MachineInstr &MI, MachineInstr *V0Def) {
#define CASE_VMERGE_TO_VMV(lmul)                                               \
  case RISCV::PseudoVMERGE_VVM_##lmul:                                         \
    NewOpc = RISCV::PseudoVMV_V_V_##lmul;                                      \
    break;
  unsigned NewOpc;
  switch (MI.getOpcode()) {
  default:
    return false;
    CASE_VMERGE_TO_VMV(MF8)
    CASE_VMERGE_TO_VMV(MF4)
    CASE_VMERGE_TO_VMV(MF2)
    CASE_VMERGE_TO_VMV(M1)
    CASE_VMERGE_TO_VMV(M2)
    CASE_VMERGE_TO_VMV(M4)
    CASE_VMERGE_TO_VMV(M8)
  }

  Register MergeReg = MI.getOperand(1).getReg();
  Register FalseReg = MI.getOperand(2).getReg();
  // Check merge == false (or merge == undef)
  if (MergeReg != RISCV::NoRegister && TRI->lookThruCopyLike(MergeReg, MRI) !=
                                           TRI->lookThruCopyLike(FalseReg, MRI))
    return false;

  assert(MI.getOperand(4).isReg() && MI.getOperand(4).getReg() == RISCV::V0);
  if (!isAllOnesMask(V0Def))
    return false;

  MI.setDesc(TII->get(NewOpc));
  MI.removeOperand(1);  // Merge operand
  MI.tieOperands(0, 1); // Tie false to dest
  MI.removeOperand(3);  // Mask operand
  MI.addOperand(
      MachineOperand::CreateImm(RISCVII::TAIL_UNDISTURBED_MASK_UNDISTURBED));

  // vmv.v.v doesn't have a mask operand, so we may be able to inflate the
  // register class for the destination and merge operands e.g. VRNoV0 -> VR
  MRI->recomputeRegClass(MI.getOperand(0).getReg());
  MRI->recomputeRegClass(MI.getOperand(1).getReg());
  return true;
}

bool RISCVFoldMasks::convertToUnmasked(MachineInstr &MI,
                                       MachineInstr *MaskDef) {
  const RISCV::RISCVMaskedPseudoInfo *I =
      RISCV::getMaskedPseudoInfo(MI.getOpcode());
  if (!I)
    return false;

  if (!isAllOnesMask(MaskDef))
    return false;

  // There are two classes of pseudos in the table - compares and
  // everything else.  See the comment on RISCVMaskedPseudo for details.
  const unsigned Opc = I->UnmaskedPseudo;
  const MCInstrDesc &MCID = TII->get(Opc);
  const bool HasPolicyOp = RISCVII::hasVecPolicyOp(MCID.TSFlags);
  const bool HasPassthru = RISCVII::isFirstDefTiedToFirstUse(MCID);
#ifndef NDEBUG
  const MCInstrDesc &MaskedMCID = TII->get(MI.getOpcode());
  assert(RISCVII::hasVecPolicyOp(MaskedMCID.TSFlags) ==
             RISCVII::hasVecPolicyOp(MCID.TSFlags) &&
         "Masked and unmasked pseudos are inconsistent");
  assert(HasPolicyOp == HasPassthru && "Unexpected pseudo structure");
#endif
  (void)HasPolicyOp;

  MI.setDesc(MCID);

  // TODO: Increment all MaskOpIdxs in tablegen by num of explicit defs?
  unsigned MaskOpIdx = I->MaskOpIdx + MI.getNumExplicitDefs();
  MI.removeOperand(MaskOpIdx);

  // The unmasked pseudo will no longer be constrained to the vrnov0 reg class,
  // so try and relax it to vr.
  MRI->recomputeRegClass(MI.getOperand(0).getReg());
  unsigned PassthruOpIdx = MI.getNumExplicitDefs();
  if (HasPassthru) {
    if (MI.getOperand(PassthruOpIdx).getReg() != RISCV::NoRegister)
      MRI->recomputeRegClass(MI.getOperand(PassthruOpIdx).getReg());
  } else
    MI.removeOperand(PassthruOpIdx);

  return true;
}

bool RISCVFoldMasks::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVInstructions())
    return false;

  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MRI->getTargetRegisterInfo();

  bool Changed = false;

  // Masked pseudos coming out of isel will have their mask operand in the form:
  //
  // $v0:vr = COPY %mask:vr
  // %x:vr = Pseudo_MASK %a:vr, %b:br, $v0:vr
  //
  // Because $v0 isn't in SSA, keep track of it so we can check the mask operand
  // on each pseudo.
  MachineInstr *CurrentV0Def;
  for (MachineBasicBlock &MBB : MF) {
    CurrentV0Def = nullptr;
    for (MachineInstr &MI : MBB) {
      Changed |= convertToUnmasked(MI, CurrentV0Def);
      Changed |= convertVMergeToVMv(MI, CurrentV0Def);

      if (MI.definesRegister(RISCV::V0, TRI))
        CurrentV0Def = &MI;
    }
  }

  return Changed;
}

FunctionPass *llvm::createRISCVFoldMasksPass() { return new RISCVFoldMasks(); }
