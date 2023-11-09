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
// It performs the following transforms:
//
// %true = PseudoFOO %passthru, ..., %vl, %sew
// %x = PseudoVMERGE_VVM %passthru, %passthru, %true, %mask, %vl, %sew
// ->
// %x = PseudoFOO_MASK %false, ..., %mask, %vl, %sew
//
// %x = PseudoFOO_MASK ..., %allonesmask, %vl, %sew
// ->
// %x = PseudoFOO ..., %vl, %sew
//
// %x = PseudoVMERGE_VVM %false, %false, %true, %allonesmask, %vl, %sew
// ->
// %x = PseudoVMV_V_V %false, %true, %vl, %sew
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVISelDAGToDAG.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallSet.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-fold-masks"

namespace {

class RISCVFoldMasks : public MachineFunctionPass {
public:
  static char ID;
  const TargetInstrInfo *TII;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  RISCVFoldMasks() : MachineFunctionPass(ID) {
    initializeRISCVFoldMasksPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  StringRef getPassName() const override { return "RISC-V Fold Masks"; }

private:
  bool foldVMergeIntoOps(MachineInstr &MI, MachineInstr *MaskDef);
  bool convertVMergeToVMv(MachineInstr &MI, MachineInstr *MaskDef);
  bool convertToUnmasked(MachineInstr &MI, MachineInstr *MaskDef);

  bool isAllOnesMask(MachineInstr *MaskDef);
  bool isOpSameAs(const MachineOperand &LHS, const MachineOperand &RHS);
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

static unsigned getVMSetForLMul(RISCVII::VLMUL LMUL) {
  switch (LMUL) {
  case RISCVII::LMUL_F8:
    return RISCV::PseudoVMSET_M_B1;
  case RISCVII::LMUL_F4:
    return RISCV::PseudoVMSET_M_B2;
  case RISCVII::LMUL_F2:
    return RISCV::PseudoVMSET_M_B4;
  case RISCVII::LMUL_1:
    return RISCV::PseudoVMSET_M_B8;
  case RISCVII::LMUL_2:
    return RISCV::PseudoVMSET_M_B16;
  case RISCVII::LMUL_4:
    return RISCV::PseudoVMSET_M_B32;
  case RISCVII::LMUL_8:
    return RISCV::PseudoVMSET_M_B64;
  case RISCVII::LMUL_RESERVED:
    llvm_unreachable("Unexpected LMUL");
  }
  llvm_unreachable("Unknown VLMUL enum");
}

// Try to sink From to before To, also sinking any instructions between From and
// To where there is a write-after-read dependency on a physical register.
static bool sinkInstructionAndDeps(MachineInstr &From, MachineInstr &To) {
  assert(From.getParent() == To.getParent());
  SmallVector<MachineInstr *> Worklist, ToSink;
  Worklist.push_back(&From);

  // Rather than compute whether or not we saw a store for every instruction,
  // just compute it once even if it's more conservative.
  bool SawStore = false;
  for (MachineBasicBlock::instr_iterator II = From.getIterator();
       II != To.getIterator(); II++) {
    if (II->mayStore()) {
      SawStore = true;
      break;
    }
  }

  while (!Worklist.empty()) {
    MachineInstr *MI = Worklist.pop_back_val();

    if (!MI->isSafeToMove(nullptr, SawStore))
      return false;

    SmallSet<Register, 8> Defs, Uses;
    for (MachineOperand &Def : MI->all_defs())
      Defs.insert(Def.getReg());
    for (MachineOperand &Use : MI->all_uses())
      Uses.insert(Use.getReg());

    // If anything from [MI, To] uses a definition of MI, we can't sink it.
    for (MachineBasicBlock::instr_iterator II = MI->getIterator();
         II != To.getIterator(); II++) {
      for (MachineOperand &Use : II->all_uses()) {
        if (Defs.contains(Use.getReg()))
          return false;
      }
    }

    // If MI uses any physical registers, we need to sink any instructions after
    // it where there might be a write-after-read dependency.
    for (MachineBasicBlock::instr_iterator II = MI->getIterator();
         II != To.getIterator(); II++) {
      bool NeedsSink = any_of(II->all_defs(), [&Uses](MachineOperand &Def) {
        return Def.getReg().isPhysical() && Uses.contains(Def.getReg());
      });
      if (NeedsSink)
        Worklist.push_back(&*II);
    }

    ToSink.push_back(MI);
  }

  for (MachineInstr *MI : ToSink)
    MI->moveBefore(&To);
  return true;
}

// Returns true if LHS is the same register as RHS, or if LHS is undefined.
bool RISCVFoldMasks::isOpSameAs(const MachineOperand &LHS,
                                const MachineOperand &RHS) {
  if (LHS.getReg() == RISCV::NoRegister)
    return true;
  if (RHS.getReg() == RISCV::NoRegister)
    return false;
  return TRI->lookThruCopyLike(LHS.getReg(), MRI) ==
         TRI->lookThruCopyLike(RHS.getReg(), MRI);
}

// Try to fold away VMERGE_VVM instructions. We handle these cases:
// -Masked TU VMERGE_VVM combined with an unmasked TA instruction instruction
//  folds to a masked TU instruction. VMERGE_VVM must have have merge operand
//  same as false operand.
// -Masked TA VMERGE_VVM combined with an unmasked TA instruction fold to a
//  masked TA instruction.
// -Unmasked TU VMERGE_VVM combined with a masked MU TA instruction folds to
//  masked TU instruction. Both instructions must have the same merge operand.
//  VMERGE_VVM must have have merge operand same as false operand.
// Note: The VMERGE_VVM forms above (TA, and TU) refer to the policy implied,
// not the pseudo name.  That is, a TA VMERGE_VVM can be either the _TU pseudo
// form with an IMPLICIT_DEF passthrough operand or the unsuffixed (TA) pseudo
// form.
bool RISCVFoldMasks::foldVMergeIntoOps(MachineInstr &MI,
                                       MachineInstr *MaskDef) {
  MachineOperand *True;
  MachineOperand *Merge;
  MachineOperand *False;

  const unsigned BaseOpc = RISCV::getRVVMCOpcode(MI.getOpcode());
  // A vmv.v.v is equivalent to a vmerge with an all-ones mask.
  if (BaseOpc == RISCV::VMV_V_V) {
    Merge = &MI.getOperand(1);
    False = &MI.getOperand(1);
    True = &MI.getOperand(2);
  } else if (BaseOpc == RISCV::VMERGE_VVM) {
    Merge = &MI.getOperand(1);
    False = &MI.getOperand(2);
    True = &MI.getOperand(3);
  } else
    return false;

  MachineInstr &TrueMI = *MRI->getVRegDef(True->getReg());

  // We require that either merge and false are the same, or that merge
  // is undefined.
  if (!isOpSameAs(*Merge, *False))
    return false;

  // N must be the only user of True.
  if (!MRI->hasOneUse(True->getReg()))
    return false;

  unsigned TrueOpc = TrueMI.getOpcode();
  const MCInstrDesc &TrueMCID = TrueMI.getDesc();
  bool HasTiedDest = RISCVII::isFirstDefTiedToFirstUse(TrueMCID);

  bool IsMasked = false;
  const RISCV::RISCVMaskedPseudoInfo *Info =
      RISCV::lookupMaskedIntrinsicByUnmasked(TrueOpc);
  if (!Info && HasTiedDest) {
    Info = RISCV::getMaskedPseudoInfo(TrueOpc);
    IsMasked = true;
  }

  if (!Info)
    return false;

  // When Mask is not a true mask, this transformation is illegal for some
  // operations whose results are affected by mask, like viota.m.
  if (Info->MaskAffectsResult && BaseOpc == RISCV::VMERGE_VVM &&
      !isAllOnesMask(MaskDef))
    return false;

  MachineOperand &TrueMergeOp = TrueMI.getOperand(1);
  if (HasTiedDest && TrueMergeOp.getReg() != RISCV::NoRegister) {
    // The vmerge instruction must be TU.
    // FIXME: This could be relaxed, but we need to handle the policy for the
    // resulting op correctly.
    if (Merge->getReg() == RISCV::NoRegister)
      return false;
    // Both the vmerge instruction and the True instruction must have the same
    // merge operand.
    if (!isOpSameAs(TrueMergeOp, *False))
      return false;
  }

  if (IsMasked) {
    assert(HasTiedDest && "Expected tied dest");
    // The vmerge instruction must be TU.
    if (Merge->getReg() == RISCV::NoRegister)
      return false;
    // The vmerge instruction must have an all 1s mask since we're going to keep
    // the mask from the True instruction.
    // FIXME: Support mask agnostic True instruction which would have an
    // undef merge operand.
    if (BaseOpc == RISCV::VMERGE_VVM && !isAllOnesMask(MaskDef))
      return false;
  }

  // Skip if True has side effect.
  // TODO: Support vleff and vlsegff.
  if (TII->get(TrueOpc).hasUnmodeledSideEffects())
    return false;

  // The vector policy operand may be present for masked intrinsics
  const MachineOperand &TrueVL =
      TrueMI.getOperand(RISCVII::getVLOpNum(TrueMCID));

  auto GetMinVL =
      [](const MachineOperand &LHS,
         const MachineOperand &RHS) -> std::optional<MachineOperand> {
    if (LHS.isReg() && RHS.isReg() && LHS.getReg().isVirtual() &&
        LHS.getReg() == RHS.getReg())
      return LHS;
    if (LHS.isImm() && LHS.getImm() == RISCV::VLMaxSentinel)
      return RHS;
    if (RHS.isImm() && RHS.getImm() == RISCV::VLMaxSentinel)
      return LHS;
    if (!LHS.isImm() || !RHS.isImm())
      return std::nullopt;
    return LHS.getImm() <= RHS.getImm() ? LHS : RHS;
  };

  // Because MI and True must have the same merge operand (or True's operand is
  // implicit_def), the "effective" body is the minimum of their VLs.
  const MachineOperand VL = MI.getOperand(RISCVII::getVLOpNum(MI.getDesc()));
  auto MinVL = GetMinVL(TrueVL, VL);
  if (!MinVL)
    return false;
  bool VLChanged = !MinVL->isIdenticalTo(VL);

  // If we end up changing the VL or mask of True, then we need to make sure it
  // doesn't raise any observable fp exceptions, since changing the active
  // elements will affect how fflags is set.
  if (VLChanged || !IsMasked)
    if (TrueMCID.mayRaiseFPException() &&
        !TrueMI.getFlag(MachineInstr::MIFlag::NoFPExcept))
      return false;

  unsigned MaskedOpc = Info->MaskedPseudo;
  const MCInstrDesc &MaskedMCID = TII->get(MaskedOpc);
#ifndef NDEBUG
  assert(RISCVII::hasVecPolicyOp(MaskedMCID.TSFlags) &&
         "Expected instructions with mask have policy operand.");
  assert(MaskedMCID.getOperandConstraint(MaskedMCID.getNumDefs(),
                                         MCOI::TIED_TO) == 0 &&
         "Expected instructions with mask have a tied dest.");
#endif

  // Sink True down to MI so that it can access MI's operands.
  if (!sinkInstructionAndDeps(TrueMI, MI))
    return false;

  // Set the merge to the false operand of the merge.
  TrueMI.getOperand(1).setReg(False->getReg());

  // If we're converting it to a masked pseudo, reuse MI's mask.
  if (!IsMasked) {
    if (BaseOpc == RISCV::VMV_V_V) {
      // If MI is a vmv.v.v, it won't have a mask operand. So insert an all-ones
      // mask just before True.
      unsigned VMSetOpc =
          getVMSetForLMul(RISCVII::getLMul(MI.getDesc().TSFlags));
      Register Dest = MRI->createVirtualRegister(&RISCV::VRRegClass);
      BuildMI(*MI.getParent(), TrueMI, MI.getDebugLoc(), TII->get(VMSetOpc),
              Dest)
          .add(VL)
          .add(TrueMI.getOperand(RISCVII::getSEWOpNum(TrueMCID)));
      BuildMI(*MI.getParent(), TrueMI, MI.getDebugLoc(), TII->get(RISCV::COPY),
              RISCV::V0)
          .addReg(Dest);
    }

    TrueMI.setDesc(MaskedMCID);

    // TODO: Increment MaskOpIdx by number of explicit defs in tablegen?
    unsigned MaskOpIdx = Info->MaskOpIdx + TrueMI.getNumExplicitDefs();
    TrueMI.insert(&TrueMI.getOperand(MaskOpIdx),
                  MachineOperand::CreateReg(RISCV::V0, false));
  }

  // Update the AVL.
  if (MinVL->isReg())
    TrueMI.getOperand(RISCVII::getVLOpNum(MaskedMCID))
        .ChangeToRegister(MinVL->getReg(), false);
  else
    TrueMI.getOperand(RISCVII::getVLOpNum(MaskedMCID))
        .ChangeToImmediate(MinVL->getImm());

  // Use a tumu policy, relaxing it to tail agnostic provided that the merge
  // operand is undefined.
  //
  // However, if the VL became smaller than what the vmerge had originally, then
  // elements past VL that were previously in the vmerge's body will have moved
  // to the tail. In that case we always need to use tail undisturbed to
  // preserve them.
  uint64_t Policy = (Merge->getReg() == RISCV::NoRegister && !VLChanged)
                        ? RISCVII::TAIL_AGNOSTIC
                        : RISCVII::TAIL_UNDISTURBED_MASK_UNDISTURBED;
  TrueMI.getOperand(RISCVII::getVecPolicyOpNum(MaskedMCID)).setImm(Policy);

  const TargetRegisterClass *V0RC =
      TII->getRegClass(MaskedMCID, 0, TRI, *MI.getMF());

  // The destination and passthru can no longer be in V0.
  MRI->constrainRegClass(TrueMI.getOperand(0).getReg(), V0RC);
  Register PassthruReg = TrueMI.getOperand(1).getReg();
  if (PassthruReg != RISCV::NoRegister)
    MRI->constrainRegClass(PassthruReg, V0RC);

  MRI->replaceRegWith(MI.getOperand(0).getReg(), TrueMI.getOperand(0).getReg());
  MI.eraseFromParent();

  return true;
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

  // Check merge == false (or merge == undef)
  if (!isOpSameAs(MI.getOperand(1), MI.getOperand(2)))
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
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      Changed |= foldVMergeIntoOps(MI, CurrentV0Def);
      if (MI.definesRegister(RISCV::V0, TRI))
        CurrentV0Def = &MI;
    }

    CurrentV0Def = nullptr;
    for (MachineInstr &MI : MBB) {
      Changed |= convertVMergeToVMv(MI, CurrentV0Def);
      Changed |= convertToUnmasked(MI, CurrentV0Def);
      if (MI.definesRegister(RISCV::V0, TRI))
        CurrentV0Def = &MI;
    }
  }

  return Changed;
}

FunctionPass *llvm::createRISCVFoldMasksPass() { return new RISCVFoldMasks(); }
