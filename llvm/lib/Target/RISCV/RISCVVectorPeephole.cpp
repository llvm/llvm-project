//===- RISCVVectorPeephole.cpp - MI Vector Pseudo Peepholes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass performs various vector pseudo peephole optimisations after
// instruction selection.
//
// Currently it converts vmerge.vvm to vmv.v.v
// PseudoVMERGE_VVM %false, %false, %true, %allonesmask, %vl, %sew
// ->
// PseudoVMV_V_V %false, %true, %vl, %sew
//
// And masked pseudos to unmasked pseudos
// PseudoVADD_V_V_MASK %passthru, %a, %b, %allonesmask, %vl, sew, policy
// ->
// PseudoVADD_V_V %passthru %a, %b, %vl, sew, policy
//
// It also converts AVLs to VLMAX where possible
// %vl = VLENB * something
// PseudoVADD_V_V %passthru, %a, %b, %vl, sew, policy
// ->
// PseudoVADD_V_V %passthru, %a, %b, -1, sew, policy
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVISelDAGToDAG.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-vector-peephole"

namespace {

class RISCVVectorPeephole : public MachineFunctionPass {
public:
  static char ID;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  const RISCVSubtarget *ST;
  RISCVVectorPeephole() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  StringRef getPassName() const override {
    return "RISC-V Vector Peephole Optimization";
  }

private:
  bool convertToVLMAX(MachineInstr &MI) const;
  bool convertToWholeRegister(MachineInstr &MI) const;
  bool convertToUnmasked(MachineInstr &MI) const;
  bool convertVMergeToVMv(MachineInstr &MI) const;

  bool isAllOnesMask(const MachineInstr *MaskDef) const;
  std::optional<unsigned> getConstant(const MachineOperand &VL) const;

  /// Maps uses of V0 to the corresponding def of V0.
  DenseMap<const MachineInstr *, const MachineInstr *> V0Defs;
};

} // namespace

char RISCVVectorPeephole::ID = 0;

INITIALIZE_PASS(RISCVVectorPeephole, DEBUG_TYPE, "RISC-V Fold Masks", false,
                false)

/// Check if an operand is an immediate or a materialized ADDI $x0, imm.
std::optional<unsigned>
RISCVVectorPeephole::getConstant(const MachineOperand &VL) const {
  if (VL.isImm())
    return VL.getImm();

  MachineInstr *Def = MRI->getVRegDef(VL.getReg());
  if (!Def || Def->getOpcode() != RISCV::ADDI ||
      Def->getOperand(1).getReg() != RISCV::X0)
    return std::nullopt;
  return Def->getOperand(2).getImm();
}

/// Convert AVLs that are known to be VLMAX to the VLMAX sentinel.
bool RISCVVectorPeephole::convertToVLMAX(MachineInstr &MI) const {
  if (!RISCVII::hasVLOp(MI.getDesc().TSFlags) ||
      !RISCVII::hasSEWOp(MI.getDesc().TSFlags))
    return false;

  auto LMUL = RISCVVType::decodeVLMUL(RISCVII::getLMul(MI.getDesc().TSFlags));
  // Fixed-point value, denominator=8
  unsigned LMULFixed = LMUL.second ? (8 / LMUL.first) : 8 * LMUL.first;
  unsigned Log2SEW = MI.getOperand(RISCVII::getSEWOpNum(MI.getDesc())).getImm();
  // A Log2SEW of 0 is an operation on mask registers only
  unsigned SEW = Log2SEW ? 1 << Log2SEW : 8;
  assert(RISCVVType::isValidSEW(SEW) && "Unexpected SEW");
  assert(8 * LMULFixed / SEW > 0);

  // If the exact VLEN is known then we know VLMAX, check if the AVL == VLMAX.
  MachineOperand &VL = MI.getOperand(RISCVII::getVLOpNum(MI.getDesc()));
  if (auto VLen = ST->getRealVLen(), AVL = getConstant(VL);
      VLen && AVL && (*VLen * LMULFixed) / SEW == *AVL * 8) {
    VL.ChangeToImmediate(RISCV::VLMaxSentinel);
    return true;
  }

  // If an AVL is a VLENB that's possibly scaled to be equal to VLMAX, convert
  // it to the VLMAX sentinel value.
  if (!VL.isReg())
    return false;
  MachineInstr *Def = MRI->getVRegDef(VL.getReg());
  if (!Def)
    return false;

  // Fixed-point value, denominator=8
  uint64_t ScaleFixed = 8;
  // Check if the VLENB was potentially scaled with slli/srli
  if (Def->getOpcode() == RISCV::SLLI) {
    assert(Def->getOperand(2).getImm() < 64);
    ScaleFixed <<= Def->getOperand(2).getImm();
    Def = MRI->getVRegDef(Def->getOperand(1).getReg());
  } else if (Def->getOpcode() == RISCV::SRLI) {
    assert(Def->getOperand(2).getImm() < 64);
    ScaleFixed >>= Def->getOperand(2).getImm();
    Def = MRI->getVRegDef(Def->getOperand(1).getReg());
  }

  if (!Def || Def->getOpcode() != RISCV::PseudoReadVLENB)
    return false;

  // AVL = (VLENB * Scale)
  //
  // VLMAX = (VLENB * 8 * LMUL) / SEW
  //
  // AVL == VLMAX
  // -> VLENB * Scale == (VLENB * 8 * LMUL) / SEW
  // -> Scale == (8 * LMUL) / SEW
  if (ScaleFixed != 8 * LMULFixed / SEW)
    return false;

  VL.ChangeToImmediate(RISCV::VLMaxSentinel);

  return true;
}

bool RISCVVectorPeephole::isAllOnesMask(const MachineInstr *MaskDef) const {
  assert(MaskDef && MaskDef->isCopy() &&
         MaskDef->getOperand(0).getReg() == RISCV::V0);
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

/// Convert unit strided unmasked loads and stores to whole-register equivalents
/// to avoid the dependency on $vl and $vtype.
///
/// %x = PseudoVLE8_V_M1 %passthru, %ptr, %vlmax, policy
/// PseudoVSE8_V_M1 %v, %ptr, %vlmax
///
/// ->
///
/// %x = VL1RE8_V %ptr
/// VS1R_V %v, %ptr
bool RISCVVectorPeephole::convertToWholeRegister(MachineInstr &MI) const {
#define CASE_WHOLE_REGISTER_LMUL_SEW(lmul, sew)                                \
  case RISCV::PseudoVLE##sew##_V_M##lmul:                                      \
    NewOpc = RISCV::VL##lmul##RE##sew##_V;                                     \
    break;                                                                     \
  case RISCV::PseudoVSE##sew##_V_M##lmul:                                      \
    NewOpc = RISCV::VS##lmul##R_V;                                             \
    break;
#define CASE_WHOLE_REGISTER_LMUL(lmul)                                         \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 8)                                        \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 16)                                       \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 32)                                       \
  CASE_WHOLE_REGISTER_LMUL_SEW(lmul, 64)

  unsigned NewOpc;
  switch (MI.getOpcode()) {
    CASE_WHOLE_REGISTER_LMUL(1)
    CASE_WHOLE_REGISTER_LMUL(2)
    CASE_WHOLE_REGISTER_LMUL(4)
    CASE_WHOLE_REGISTER_LMUL(8)
  default:
    return false;
  }

  MachineOperand &VLOp = MI.getOperand(RISCVII::getVLOpNum(MI.getDesc()));
  if (!VLOp.isImm() || VLOp.getImm() != RISCV::VLMaxSentinel)
    return false;

  // Whole register instructions aren't pseudos so they don't have
  // policy/SEW/AVL ops, and they don't have passthrus.
  if (RISCVII::hasVecPolicyOp(MI.getDesc().TSFlags))
    MI.removeOperand(RISCVII::getVecPolicyOpNum(MI.getDesc()));
  MI.removeOperand(RISCVII::getSEWOpNum(MI.getDesc()));
  MI.removeOperand(RISCVII::getVLOpNum(MI.getDesc()));
  if (RISCVII::isFirstDefTiedToFirstUse(MI.getDesc()))
    MI.removeOperand(1);

  MI.setDesc(TII->get(NewOpc));

  return true;
}

// Transform (VMERGE_VVM_<LMUL> false, false, true, allones, vl, sew) to
// (VMV_V_V_<LMUL> false, true, vl, sew). It may decrease uses of VMSET.
bool RISCVVectorPeephole::convertVMergeToVMv(MachineInstr &MI) const {
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
  if (!isAllOnesMask(V0Defs.lookup(&MI)))
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

bool RISCVVectorPeephole::convertToUnmasked(MachineInstr &MI) const {
  const RISCV::RISCVMaskedPseudoInfo *I =
      RISCV::getMaskedPseudoInfo(MI.getOpcode());
  if (!I)
    return false;

  if (!isAllOnesMask(V0Defs.lookup(&MI)))
    return false;

  // There are two classes of pseudos in the table - compares and
  // everything else.  See the comment on RISCVMaskedPseudo for details.
  const unsigned Opc = I->UnmaskedPseudo;
  const MCInstrDesc &MCID = TII->get(Opc);
  [[maybe_unused]] const bool HasPolicyOp =
      RISCVII::hasVecPolicyOp(MCID.TSFlags);
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

bool RISCVVectorPeephole::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  // Skip if the vector extension is not enabled.
  ST = &MF.getSubtarget<RISCVSubtarget>();
  if (!ST->hasVInstructions())
    return false;

  TII = ST->getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = MRI->getTargetRegisterInfo();

  bool Changed = false;

  // Masked pseudos coming out of isel will have their mask operand in the form:
  //
  // $v0:vr = COPY %mask:vr
  // %x:vr = Pseudo_MASK %a:vr, %b:br, $v0:vr
  //
  // Because $v0 isn't in SSA, keep track of its definition at each use so we
  // can check mask operands.
  for (const MachineBasicBlock &MBB : MF) {
    const MachineInstr *CurrentV0Def = nullptr;
    for (const MachineInstr &MI : MBB) {
      if (MI.readsRegister(RISCV::V0, TRI))
        V0Defs[&MI] = CurrentV0Def;

      if (MI.definesRegister(RISCV::V0, TRI))
        CurrentV0Def = &MI;
    }
  }

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      Changed |= convertToVLMAX(MI);
      Changed |= convertToUnmasked(MI);
      Changed |= convertToWholeRegister(MI);
      Changed |= convertVMergeToVMv(MI);
    }
  }

  return Changed;
}

FunctionPass *llvm::createRISCVVectorPeepholePass() {
  return new RISCVVectorPeephole();
}
