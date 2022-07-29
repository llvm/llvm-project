//===-- SIFoldOperands.cpp - Fold operands --- ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//
//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

#define DEBUG_TYPE "si-fold-operands"
using namespace llvm;

namespace {

struct FoldCandidate {
  MachineInstr *UseMI;
  union {
    MachineOperand *OpToFold;
    uint64_t ImmToFold;
    int FrameIndexToFold;
  };
  int ShrinkOpcode;
  unsigned UseOpNo;
  MachineOperand::MachineOperandType Kind;
  bool Commuted;

  FoldCandidate(MachineInstr *MI, unsigned OpNo, MachineOperand *FoldOp,
                bool Commuted_ = false,
                int ShrinkOp = -1) :
    UseMI(MI), OpToFold(nullptr), ShrinkOpcode(ShrinkOp), UseOpNo(OpNo),
    Kind(FoldOp->getType()),
    Commuted(Commuted_) {
    if (FoldOp->isImm()) {
      ImmToFold = FoldOp->getImm();
    } else if (FoldOp->isFI()) {
      FrameIndexToFold = FoldOp->getIndex();
    } else {
      assert(FoldOp->isReg() || FoldOp->isGlobal());
      OpToFold = FoldOp;
    }
  }

  bool isFI() const {
    return Kind == MachineOperand::MO_FrameIndex;
  }

  bool isImm() const {
    return Kind == MachineOperand::MO_Immediate;
  }

  bool isReg() const {
    return Kind == MachineOperand::MO_Register;
  }

  bool isGlobal() const { return Kind == MachineOperand::MO_GlobalAddress; }

  bool isCommuted() const {
    return Commuted;
  }

  bool needsShrink() const {
    return ShrinkOpcode != -1;
  }

  int getShrinkOpcode() const {
    return ShrinkOpcode;
  }
};

class SIFoldOperands : public MachineFunctionPass {
public:
  static char ID;
  MachineRegisterInfo *MRI;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  const GCNSubtarget *ST;
  const SIMachineFunctionInfo *MFI;

  void foldOperand(MachineOperand &OpToFold,
                   MachineInstr *UseMI,
                   int UseOpIdx,
                   SmallVectorImpl<FoldCandidate> &FoldList,
                   SmallVectorImpl<MachineInstr *> &CopiesToReplace) const;

  bool tryFoldCndMask(MachineInstr &MI) const;
  bool tryFoldZeroHighBits(MachineInstr &MI) const;
  bool foldInstOperand(MachineInstr &MI, MachineOperand &OpToFold) const;

  const MachineOperand *isClamp(const MachineInstr &MI) const;
  bool tryFoldClamp(MachineInstr &MI);

  std::pair<const MachineOperand *, int> isOMod(const MachineInstr &MI) const;
  bool tryFoldOMod(MachineInstr &MI);
  bool tryFoldRegSequence(MachineInstr &MI);
  bool tryFoldLCSSAPhi(MachineInstr &MI);
  bool tryFoldLoad(MachineInstr &MI);

public:
  SIFoldOperands() : MachineFunctionPass(ID) {
    initializeSIFoldOperandsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI Fold Operands"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(SIFoldOperands, DEBUG_TYPE,
                "SI Fold Operands", false, false)

char SIFoldOperands::ID = 0;

char &llvm::SIFoldOperandsID = SIFoldOperands::ID;

// Map multiply-accumulate opcode to corresponding multiply-add opcode if any.
static unsigned macToMad(unsigned Opc) {
  switch (Opc) {
  case AMDGPU::V_MAC_F32_e64:
    return AMDGPU::V_MAD_F32_e64;
  case AMDGPU::V_MAC_F16_e64:
    return AMDGPU::V_MAD_F16_e64;
  case AMDGPU::V_FMAC_F32_e64:
    return AMDGPU::V_FMA_F32_e64;
  case AMDGPU::V_FMAC_F16_e64:
    return AMDGPU::V_FMA_F16_gfx9_e64;
  case AMDGPU::V_FMAC_LEGACY_F32_e64:
    return AMDGPU::V_FMA_LEGACY_F32_e64;
  case AMDGPU::V_FMAC_F64_e64:
    return AMDGPU::V_FMA_F64_e64;
  }
  return AMDGPU::INSTRUCTION_LIST_END;
}

// TODO: Add heuristic that the frame index might not fit in the addressing mode
// immediate offset to avoid materializing in loops.
static bool frameIndexMayFold(const SIInstrInfo *TII,
                              const MachineInstr &UseMI,
                              int OpNo,
                              const MachineOperand &OpToFold) {
  if (!OpToFold.isFI())
    return false;

  if (TII->isMUBUF(UseMI))
    return OpNo == AMDGPU::getNamedOperandIdx(UseMI.getOpcode(),
                                              AMDGPU::OpName::vaddr);
  if (!TII->isFLATScratch(UseMI))
    return false;

  int SIdx = AMDGPU::getNamedOperandIdx(UseMI.getOpcode(),
                                        AMDGPU::OpName::saddr);
  if (OpNo == SIdx)
    return true;

  int VIdx = AMDGPU::getNamedOperandIdx(UseMI.getOpcode(),
                                        AMDGPU::OpName::vaddr);
  return OpNo == VIdx && SIdx == -1;
}

FunctionPass *llvm::createSIFoldOperandsPass() {
  return new SIFoldOperands();
}

static bool updateOperand(FoldCandidate &Fold,
                          const SIInstrInfo &TII,
                          const TargetRegisterInfo &TRI,
                          const GCNSubtarget &ST) {
  MachineInstr *MI = Fold.UseMI;
  MachineOperand &Old = MI->getOperand(Fold.UseOpNo);
  assert(Old.isReg());

  if (Fold.isImm()) {
    if (MI->getDesc().TSFlags & SIInstrFlags::IsPacked &&
        !(MI->getDesc().TSFlags & SIInstrFlags::IsMAI) &&
        (!ST.hasDOTOpSelHazard() ||
         !(MI->getDesc().TSFlags & SIInstrFlags::IsDOT)) &&
        AMDGPU::isFoldableLiteralV216(Fold.ImmToFold,
                                      ST.hasInv2PiInlineImm())) {
      // Set op_sel/op_sel_hi on this operand or bail out if op_sel is
      // already set.
      unsigned Opcode = MI->getOpcode();
      int OpNo = MI->getOperandNo(&Old);
      int ModIdx = -1;
      if (OpNo == AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src0))
        ModIdx = AMDGPU::OpName::src0_modifiers;
      else if (OpNo == AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src1))
        ModIdx = AMDGPU::OpName::src1_modifiers;
      else if (OpNo == AMDGPU::getNamedOperandIdx(Opcode, AMDGPU::OpName::src2))
        ModIdx = AMDGPU::OpName::src2_modifiers;
      assert(ModIdx != -1);
      ModIdx = AMDGPU::getNamedOperandIdx(Opcode, ModIdx);
      MachineOperand &Mod = MI->getOperand(ModIdx);
      unsigned Val = Mod.getImm();
      if (!(Val & SISrcMods::OP_SEL_0) && (Val & SISrcMods::OP_SEL_1)) {
        // Only apply the following transformation if that operand requires
        // a packed immediate.
        switch (TII.get(Opcode).OpInfo[OpNo].OperandType) {
        case AMDGPU::OPERAND_REG_IMM_V2FP16:
        case AMDGPU::OPERAND_REG_IMM_V2INT16:
        case AMDGPU::OPERAND_REG_INLINE_C_V2FP16:
        case AMDGPU::OPERAND_REG_INLINE_C_V2INT16:
          // If upper part is all zero we do not need op_sel_hi.
          if (!isUInt<16>(Fold.ImmToFold)) {
            if (!(Fold.ImmToFold & 0xffff)) {
              Mod.setImm(Mod.getImm() | SISrcMods::OP_SEL_0);
              Mod.setImm(Mod.getImm() & ~SISrcMods::OP_SEL_1);
              Old.ChangeToImmediate((Fold.ImmToFold >> 16) & 0xffff);
              return true;
            }
            Mod.setImm(Mod.getImm() & ~SISrcMods::OP_SEL_1);
            Old.ChangeToImmediate(Fold.ImmToFold & 0xffff);
            return true;
          }
          break;
        default:
          break;
        }
      }
    }
  }

  if ((Fold.isImm() || Fold.isFI() || Fold.isGlobal()) && Fold.needsShrink()) {
    MachineBasicBlock *MBB = MI->getParent();
    auto Liveness = MBB->computeRegisterLiveness(&TRI, AMDGPU::VCC, MI, 16);
    if (Liveness != MachineBasicBlock::LQR_Dead) {
      LLVM_DEBUG(dbgs() << "Not shrinking " << MI << " due to vcc liveness\n");
      return false;
    }

    MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
    int Op32 = Fold.getShrinkOpcode();
    MachineOperand &Dst0 = MI->getOperand(0);
    MachineOperand &Dst1 = MI->getOperand(1);
    assert(Dst0.isDef() && Dst1.isDef());

    bool HaveNonDbgCarryUse = !MRI.use_nodbg_empty(Dst1.getReg());

    const TargetRegisterClass *Dst0RC = MRI.getRegClass(Dst0.getReg());
    Register NewReg0 = MRI.createVirtualRegister(Dst0RC);

    MachineInstr *Inst32 = TII.buildShrunkInst(*MI, Op32);

    if (HaveNonDbgCarryUse) {
      BuildMI(*MBB, MI, MI->getDebugLoc(), TII.get(AMDGPU::COPY), Dst1.getReg())
        .addReg(AMDGPU::VCC, RegState::Kill);
    }

    // Keep the old instruction around to avoid breaking iterators, but
    // replace it with a dummy instruction to remove uses.
    //
    // FIXME: We should not invert how this pass looks at operands to avoid
    // this. Should track set of foldable movs instead of looking for uses
    // when looking at a use.
    Dst0.setReg(NewReg0);
    for (unsigned I = MI->getNumOperands() - 1; I > 0; --I)
      MI->removeOperand(I);
    MI->setDesc(TII.get(AMDGPU::IMPLICIT_DEF));

    if (Fold.isCommuted())
      TII.commuteInstruction(*Inst32, false);
    return true;
  }

  assert(!Fold.needsShrink() && "not handled");

  if (Fold.isImm()) {
    if (Old.isTied()) {
      int NewMFMAOpc = AMDGPU::getMFMAEarlyClobberOp(MI->getOpcode());
      if (NewMFMAOpc == -1)
        return false;
      MI->setDesc(TII.get(NewMFMAOpc));
      MI->untieRegOperand(0);
    }
    Old.ChangeToImmediate(Fold.ImmToFold);
    return true;
  }

  if (Fold.isGlobal()) {
    Old.ChangeToGA(Fold.OpToFold->getGlobal(), Fold.OpToFold->getOffset(),
                   Fold.OpToFold->getTargetFlags());
    return true;
  }

  if (Fold.isFI()) {
    Old.ChangeToFrameIndex(Fold.FrameIndexToFold);
    return true;
  }

  MachineOperand *New = Fold.OpToFold;
  Old.substVirtReg(New->getReg(), New->getSubReg(), TRI);
  Old.setIsUndef(New->isUndef());
  return true;
}

static bool isUseMIInFoldList(ArrayRef<FoldCandidate> FoldList,
                              const MachineInstr *MI) {
  for (auto Candidate : FoldList) {
    if (Candidate.UseMI == MI)
      return true;
  }
  return false;
}

static void appendFoldCandidate(SmallVectorImpl<FoldCandidate> &FoldList,
                                MachineInstr *MI, unsigned OpNo,
                                MachineOperand *FoldOp, bool Commuted = false,
                                int ShrinkOp = -1) {
  // Skip additional folding on the same operand.
  for (FoldCandidate &Fold : FoldList)
    if (Fold.UseMI == MI && Fold.UseOpNo == OpNo)
      return;
  LLVM_DEBUG(dbgs() << "Append " << (Commuted ? "commuted" : "normal")
                    << " operand " << OpNo << "\n  " << *MI);
  FoldList.emplace_back(MI, OpNo, FoldOp, Commuted, ShrinkOp);
}

static bool tryAddToFoldList(SmallVectorImpl<FoldCandidate> &FoldList,
                             MachineInstr *MI, unsigned OpNo,
                             MachineOperand *OpToFold,
                             const SIInstrInfo *TII) {
  const unsigned Opc = MI->getOpcode();

  auto tryToFoldAsFMAAKorMK = [&]() {
    if (!OpToFold->isImm())
      return false;

    const bool TryAK = OpNo == 3;
    const unsigned NewOpc = TryAK ? AMDGPU::S_FMAAK_F32 : AMDGPU::S_FMAMK_F32;
    MI->setDesc(TII->get(NewOpc));

    // We have to fold into operand which would be Imm not into OpNo.
    bool FoldAsFMAAKorMK =
        tryAddToFoldList(FoldList, MI, TryAK ? 3 : 2, OpToFold, TII);
    if (FoldAsFMAAKorMK) {
      // Untie Src2 of fmac.
      MI->untieRegOperand(3);
      // For fmamk swap operands 1 and 2 if OpToFold was meant for operand 1.
      if (OpNo == 1) {
        MachineOperand &Op1 = MI->getOperand(1);
        MachineOperand &Op2 = MI->getOperand(2);
        Register OldReg = Op1.getReg();
        // Operand 2 might be an inlinable constant
        if (Op2.isImm()) {
          Op1.ChangeToImmediate(Op2.getImm());
          Op2.ChangeToRegister(OldReg, false);
        } else {
          Op1.setReg(Op2.getReg());
          Op2.setReg(OldReg);
        }
      }
      return true;
    }
    MI->setDesc(TII->get(Opc));
    return false;
  };

  if (!TII->isOperandLegal(*MI, OpNo, OpToFold)) {
    // Special case for v_mac_{f16, f32}_e64 if we are trying to fold into src2
    unsigned NewOpc = macToMad(Opc);
    if (NewOpc != AMDGPU::INSTRUCTION_LIST_END) {
      // Check if changing this to a v_mad_{f16, f32} instruction will allow us
      // to fold the operand.
      MI->setDesc(TII->get(NewOpc));
      bool FoldAsMAD = tryAddToFoldList(FoldList, MI, OpNo, OpToFold, TII);
      if (FoldAsMAD) {
        MI->untieRegOperand(OpNo);
        return true;
      }
      MI->setDesc(TII->get(Opc));
    }

    // Special case for s_fmac_f32 if we are trying to fold into Src2.
    // By transforming into fmaak we can untie Src2 and make folding legal.
    if (Opc == AMDGPU::S_FMAC_F32 && OpNo == 3) {
      if (tryToFoldAsFMAAKorMK())
        return true;
    }

    // Special case for s_setreg_b32
    if (OpToFold->isImm()) {
      unsigned ImmOpc = 0;
      if (Opc == AMDGPU::S_SETREG_B32)
        ImmOpc = AMDGPU::S_SETREG_IMM32_B32;
      else if (Opc == AMDGPU::S_SETREG_B32_mode)
        ImmOpc = AMDGPU::S_SETREG_IMM32_B32_mode;
      if (ImmOpc) {
        MI->setDesc(TII->get(ImmOpc));
        appendFoldCandidate(FoldList, MI, OpNo, OpToFold);
        return true;
      }
    }

    // If we are already folding into another operand of MI, then
    // we can't commute the instruction, otherwise we risk making the
    // other fold illegal.
    if (isUseMIInFoldList(FoldList, MI))
      return false;

    unsigned CommuteOpNo = OpNo;

    // Operand is not legal, so try to commute the instruction to
    // see if this makes it possible to fold.
    unsigned CommuteIdx0 = TargetInstrInfo::CommuteAnyOperandIndex;
    unsigned CommuteIdx1 = TargetInstrInfo::CommuteAnyOperandIndex;
    bool CanCommute = TII->findCommutedOpIndices(*MI, CommuteIdx0, CommuteIdx1);

    if (CanCommute) {
      if (CommuteIdx0 == OpNo)
        CommuteOpNo = CommuteIdx1;
      else if (CommuteIdx1 == OpNo)
        CommuteOpNo = CommuteIdx0;
    }


    // One of operands might be an Imm operand, and OpNo may refer to it after
    // the call of commuteInstruction() below. Such situations are avoided
    // here explicitly as OpNo must be a register operand to be a candidate
    // for memory folding.
    if (CanCommute && (!MI->getOperand(CommuteIdx0).isReg() ||
                       !MI->getOperand(CommuteIdx1).isReg()))
      return false;

    if (!CanCommute ||
        !TII->commuteInstruction(*MI, false, CommuteIdx0, CommuteIdx1))
      return false;

    if (!TII->isOperandLegal(*MI, CommuteOpNo, OpToFold)) {
      if ((Opc == AMDGPU::V_ADD_CO_U32_e64 ||
           Opc == AMDGPU::V_SUB_CO_U32_e64 ||
           Opc == AMDGPU::V_SUBREV_CO_U32_e64) && // FIXME
          (OpToFold->isImm() || OpToFold->isFI() || OpToFold->isGlobal())) {
        MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

        // Verify the other operand is a VGPR, otherwise we would violate the
        // constant bus restriction.
        unsigned OtherIdx = CommuteOpNo == CommuteIdx0 ? CommuteIdx1 : CommuteIdx0;
        MachineOperand &OtherOp = MI->getOperand(OtherIdx);
        if (!OtherOp.isReg() ||
            !TII->getRegisterInfo().isVGPR(MRI, OtherOp.getReg()))
          return false;

        assert(MI->getOperand(1).isDef());

        // Make sure to get the 32-bit version of the commuted opcode.
        unsigned MaybeCommutedOpc = MI->getOpcode();
        int Op32 = AMDGPU::getVOPe32(MaybeCommutedOpc);

        appendFoldCandidate(FoldList, MI, CommuteOpNo, OpToFold, true, Op32);
        return true;
      }

      TII->commuteInstruction(*MI, false, CommuteIdx0, CommuteIdx1);
      return false;
    }

    appendFoldCandidate(FoldList, MI, CommuteOpNo, OpToFold, true);
    return true;
  }

  // Inlineable constant might have been folded into Imm operand of fmaak or
  // fmamk and we are trying to fold a non-inlinable constant.
  if ((Opc == AMDGPU::S_FMAAK_F32 || Opc == AMDGPU::S_FMAMK_F32) &&
      !TII->isInlineConstant(*OpToFold)) {
    unsigned OpImm = Opc == AMDGPU::S_FMAAK_F32 ? 3 : 2;
    if (TII->isInlineConstant(*MI, MI->getOperand(OpNo), MI->getOperand(OpImm)))
      return tryToFoldAsFMAAKorMK();
  }

  // Special case for s_fmac_f32 if we are trying to fold into Src0 or Src1.
  // By changing into fmamk we can untie Src2.
  // If folding for Src0 happens first and it is identical operand to Src1 we
  // should avoid transforming into fmamk which requires commuting as it would
  // cause folding into Src1 to fail later on due to wrong OpNo used.
  if (Opc == AMDGPU::S_FMAC_F32 &&
      (OpNo != 1 || !MI->getOperand(1).isIdenticalTo(MI->getOperand(2)))) {
    if (tryToFoldAsFMAAKorMK())
      return true;
  }

  // Check the case where we might introduce a second constant operand to a
  // scalar instruction
  if (TII->isSALU(MI->getOpcode())) {
    const MCInstrDesc &InstDesc = MI->getDesc();
    const MCOperandInfo &OpInfo = InstDesc.OpInfo[OpNo];
    const SIRegisterInfo &SRI = TII->getRegisterInfo();

    // Fine if the operand can be encoded as an inline constant
    if (TII->isLiteralConstantLike(*OpToFold, OpInfo)) {
      if (!SRI.opCanUseInlineConstant(OpInfo.OperandType) ||
          !TII->isInlineConstant(*OpToFold, OpInfo)) {
        // Otherwise check for another constant
        for (unsigned i = 0, e = InstDesc.getNumOperands(); i != e; ++i) {
          auto &Op = MI->getOperand(i);
          if (OpNo != i &&
              TII->isLiteralConstantLike(Op, InstDesc.OpInfo[i])) {
            return false;
          }
        }
      }
    }
  }

  appendFoldCandidate(FoldList, MI, OpNo, OpToFold);
  return true;
}

// If the use operand doesn't care about the value, this may be an operand only
// used for register indexing, in which case it is unsafe to fold.
static bool isUseSafeToFold(const SIInstrInfo *TII,
                            const MachineInstr &MI,
                            const MachineOperand &UseMO) {
  if (UseMO.isUndef() || TII->isSDWA(MI))
    return false;

  switch (MI.getOpcode()) {
  case AMDGPU::V_MOV_B32_e32:
  case AMDGPU::V_MOV_B32_e64:
  case AMDGPU::V_MOV_B64_PSEUDO:
  case AMDGPU::V_MOV_B64_e32:
  case AMDGPU::V_MOV_B64_e64:
    // Do not fold into an indirect mov.
    return !MI.hasRegisterImplicitUseOperand(AMDGPU::M0);
  }

  return true;
  //return !MI.hasRegisterImplicitUseOperand(UseMO.getReg());
}

// Find a def of the UseReg, check if it is a reg_sequence and find initializers
// for each subreg, tracking it to foldable inline immediate if possible.
// Returns true on success.
static bool getRegSeqInit(
    SmallVectorImpl<std::pair<MachineOperand*, unsigned>> &Defs,
    Register UseReg, uint8_t OpTy,
    const SIInstrInfo *TII, const MachineRegisterInfo &MRI) {
  MachineInstr *Def = MRI.getVRegDef(UseReg);
  if (!Def || !Def->isRegSequence())
    return false;

  for (unsigned I = 1, E = Def->getNumExplicitOperands(); I < E; I += 2) {
    MachineOperand *Sub = &Def->getOperand(I);
    assert(Sub->isReg());

    for (MachineInstr *SubDef = MRI.getVRegDef(Sub->getReg());
         SubDef && Sub->isReg() && Sub->getReg().isVirtual() &&
         !Sub->getSubReg() && TII->isFoldableCopy(*SubDef);
         SubDef = MRI.getVRegDef(Sub->getReg())) {
      MachineOperand *Op = &SubDef->getOperand(1);
      if (Op->isImm()) {
        if (TII->isInlineConstant(*Op, OpTy))
          Sub = Op;
        break;
      }
      if (!Op->isReg() || Op->getReg().isPhysical())
        break;
      Sub = Op;
    }

    Defs.emplace_back(Sub, Def->getOperand(I + 1).getImm());
  }

  return true;
}

static bool tryToFoldACImm(const SIInstrInfo *TII,
                           const MachineOperand &OpToFold,
                           MachineInstr *UseMI,
                           unsigned UseOpIdx,
                           SmallVectorImpl<FoldCandidate> &FoldList) {
  const MCInstrDesc &Desc = UseMI->getDesc();
  const MCOperandInfo *OpInfo = Desc.OpInfo;
  if (!OpInfo || UseOpIdx >= Desc.getNumOperands())
    return false;

  uint8_t OpTy = OpInfo[UseOpIdx].OperandType;
  if ((OpTy < AMDGPU::OPERAND_REG_INLINE_AC_FIRST ||
       OpTy > AMDGPU::OPERAND_REG_INLINE_AC_LAST) &&
      (OpTy < AMDGPU::OPERAND_REG_INLINE_C_FIRST ||
       OpTy > AMDGPU::OPERAND_REG_INLINE_C_LAST))
    return false;

  if (OpToFold.isImm() && TII->isInlineConstant(OpToFold, OpTy) &&
      TII->isOperandLegal(*UseMI, UseOpIdx, &OpToFold)) {
    UseMI->getOperand(UseOpIdx).ChangeToImmediate(OpToFold.getImm());
    return true;
  }

  if (!OpToFold.isReg())
    return false;

  Register UseReg = OpToFold.getReg();
  if (!UseReg.isVirtual())
    return false;

  if (isUseMIInFoldList(FoldList, UseMI))
    return false;

  MachineRegisterInfo &MRI = UseMI->getParent()->getParent()->getRegInfo();

  // Maybe it is just a COPY of an immediate itself.
  MachineInstr *Def = MRI.getVRegDef(UseReg);
  MachineOperand &UseOp = UseMI->getOperand(UseOpIdx);
  if (!UseOp.getSubReg() && Def && TII->isFoldableCopy(*Def)) {
    MachineOperand &DefOp = Def->getOperand(1);
    if (DefOp.isImm() && TII->isInlineConstant(DefOp, OpTy) &&
        TII->isOperandLegal(*UseMI, UseOpIdx, &DefOp)) {
      UseMI->getOperand(UseOpIdx).ChangeToImmediate(DefOp.getImm());
      return true;
    }
  }

  SmallVector<std::pair<MachineOperand*, unsigned>, 32> Defs;
  if (!getRegSeqInit(Defs, UseReg, OpTy, TII, MRI))
    return false;

  int32_t Imm;
  for (unsigned I = 0, E = Defs.size(); I != E; ++I) {
    const MachineOperand *Op = Defs[I].first;
    if (!Op->isImm())
      return false;

    auto SubImm = Op->getImm();
    if (!I) {
      Imm = SubImm;
      if (!TII->isInlineConstant(*Op, OpTy) ||
          !TII->isOperandLegal(*UseMI, UseOpIdx, Op))
        return false;

      continue;
    }
    if (Imm != SubImm)
      return false; // Can only fold splat constants
  }

  appendFoldCandidate(FoldList, UseMI, UseOpIdx, Defs[0].first);
  return true;
}

void SIFoldOperands::foldOperand(
  MachineOperand &OpToFold,
  MachineInstr *UseMI,
  int UseOpIdx,
  SmallVectorImpl<FoldCandidate> &FoldList,
  SmallVectorImpl<MachineInstr *> &CopiesToReplace) const {
  const MachineOperand &UseOp = UseMI->getOperand(UseOpIdx);

  if (!isUseSafeToFold(TII, *UseMI, UseOp))
    return;

  // FIXME: Fold operands with subregs.
  if (UseOp.isReg() && OpToFold.isReg()) {
    if (UseOp.isImplicit() || UseOp.getSubReg() != AMDGPU::NoSubRegister)
      return;
  }

  // Special case for REG_SEQUENCE: We can't fold literals into
  // REG_SEQUENCE instructions, so we have to fold them into the
  // uses of REG_SEQUENCE.
  if (UseMI->isRegSequence()) {
    Register RegSeqDstReg = UseMI->getOperand(0).getReg();
    unsigned RegSeqDstSubReg = UseMI->getOperand(UseOpIdx + 1).getImm();

    for (auto &RSUse : make_early_inc_range(MRI->use_nodbg_operands(RegSeqDstReg))) {
      MachineInstr *RSUseMI = RSUse.getParent();

      if (tryToFoldACImm(TII, UseMI->getOperand(0), RSUseMI,
                         RSUseMI->getOperandNo(&RSUse), FoldList))
        continue;

      if (RSUse.getSubReg() != RegSeqDstSubReg)
        continue;

      foldOperand(OpToFold, RSUseMI, RSUseMI->getOperandNo(&RSUse), FoldList,
                  CopiesToReplace);
    }

    return;
  }

  if (tryToFoldACImm(TII, OpToFold, UseMI, UseOpIdx, FoldList))
    return;

  if (frameIndexMayFold(TII, *UseMI, UseOpIdx, OpToFold)) {
    // Verify that this is a stack access.
    // FIXME: Should probably use stack pseudos before frame lowering.

    if (TII->isMUBUF(*UseMI)) {
      if (TII->getNamedOperand(*UseMI, AMDGPU::OpName::srsrc)->getReg() !=
          MFI->getScratchRSrcReg())
        return;

      // Ensure this is either relative to the current frame or the current
      // wave.
      MachineOperand &SOff =
          *TII->getNamedOperand(*UseMI, AMDGPU::OpName::soffset);
      if (!SOff.isImm() || SOff.getImm() != 0)
        return;
    }

    // A frame index will resolve to a positive constant, so it should always be
    // safe to fold the addressing mode, even pre-GFX9.
    UseMI->getOperand(UseOpIdx).ChangeToFrameIndex(OpToFold.getIndex());

    if (TII->isFLATScratch(*UseMI) &&
        AMDGPU::getNamedOperandIdx(UseMI->getOpcode(),
                                   AMDGPU::OpName::vaddr) != -1 &&
        AMDGPU::getNamedOperandIdx(UseMI->getOpcode(),
                                   AMDGPU::OpName::saddr) == -1) {
      unsigned NewOpc = AMDGPU::getFlatScratchInstSSfromSV(UseMI->getOpcode());
      UseMI->setDesc(TII->get(NewOpc));
    }

    return;
  }

  bool FoldingImmLike =
      OpToFold.isImm() || OpToFold.isFI() || OpToFold.isGlobal();

  if (FoldingImmLike && UseMI->isCopy()) {
    Register DestReg = UseMI->getOperand(0).getReg();
    Register SrcReg = UseMI->getOperand(1).getReg();
    assert(SrcReg.isVirtual());

    const TargetRegisterClass *SrcRC = MRI->getRegClass(SrcReg);

    // Don't fold into a copy to a physical register with the same class. Doing
    // so would interfere with the register coalescer's logic which would avoid
    // redundant initializations.
    if (DestReg.isPhysical() && SrcRC->contains(DestReg))
      return;

    const TargetRegisterClass *DestRC = TRI->getRegClassForReg(*MRI, DestReg);
    if (!DestReg.isPhysical()) {
      if (TRI->isSGPRClass(SrcRC) && TRI->hasVectorRegisters(DestRC)) {
        SmallVector<FoldCandidate, 4> CopyUses;
        for (auto &Use : MRI->use_nodbg_operands(DestReg)) {
          // There's no point trying to fold into an implicit operand.
          if (Use.isImplicit())
            continue;

          CopyUses.emplace_back(Use.getParent(),
                                Use.getParent()->getOperandNo(&Use),
                                &UseMI->getOperand(1));
        }
        for (auto &F : CopyUses) {
          foldOperand(*F.OpToFold, F.UseMI, F.UseOpNo, FoldList, CopiesToReplace);
        }
      }

      if (DestRC == &AMDGPU::AGPR_32RegClass &&
          TII->isInlineConstant(OpToFold, AMDGPU::OPERAND_REG_INLINE_C_INT32)) {
        UseMI->setDesc(TII->get(AMDGPU::V_ACCVGPR_WRITE_B32_e64));
        UseMI->getOperand(1).ChangeToImmediate(OpToFold.getImm());
        CopiesToReplace.push_back(UseMI);
        return;
      }
    }

    // In order to fold immediates into copies, we need to change the
    // copy to a MOV.

    unsigned MovOp = TII->getMovOpcode(DestRC);
    if (MovOp == AMDGPU::COPY)
      return;

    UseMI->setDesc(TII->get(MovOp));
    MachineInstr::mop_iterator ImpOpI = UseMI->implicit_operands().begin();
    MachineInstr::mop_iterator ImpOpE = UseMI->implicit_operands().end();
    while (ImpOpI != ImpOpE) {
      MachineInstr::mop_iterator Tmp = ImpOpI;
      ImpOpI++;
      UseMI->removeOperand(UseMI->getOperandNo(Tmp));
    }
    CopiesToReplace.push_back(UseMI);
  } else {
    if (UseMI->isCopy() && OpToFold.isReg() &&
        UseMI->getOperand(0).getReg().isVirtual() &&
        !UseMI->getOperand(1).getSubReg()) {
      LLVM_DEBUG(dbgs() << "Folding " << OpToFold << "\n into " << *UseMI);
      unsigned Size = TII->getOpSize(*UseMI, 1);
      Register UseReg = OpToFold.getReg();
      UseMI->getOperand(1).setReg(UseReg);
      UseMI->getOperand(1).setSubReg(OpToFold.getSubReg());
      UseMI->getOperand(1).setIsKill(false);
      CopiesToReplace.push_back(UseMI);
      OpToFold.setIsKill(false);

      // Remove kill flags as kills may now be out of order with uses.
      MRI->clearKillFlags(OpToFold.getReg());

      // That is very tricky to store a value into an AGPR. v_accvgpr_write_b32
      // can only accept VGPR or inline immediate. Recreate a reg_sequence with
      // its initializers right here, so we will rematerialize immediates and
      // avoid copies via different reg classes.
      SmallVector<std::pair<MachineOperand*, unsigned>, 32> Defs;
      if (Size > 4 && TRI->isAGPR(*MRI, UseMI->getOperand(0).getReg()) &&
          getRegSeqInit(Defs, UseReg, AMDGPU::OPERAND_REG_INLINE_C_INT32, TII,
                        *MRI)) {
        const DebugLoc &DL = UseMI->getDebugLoc();
        MachineBasicBlock &MBB = *UseMI->getParent();

        UseMI->setDesc(TII->get(AMDGPU::REG_SEQUENCE));
        for (unsigned I = UseMI->getNumOperands() - 1; I > 0; --I)
          UseMI->removeOperand(I);

        MachineInstrBuilder B(*MBB.getParent(), UseMI);
        DenseMap<TargetInstrInfo::RegSubRegPair, Register> VGPRCopies;
        SmallSetVector<TargetInstrInfo::RegSubRegPair, 32> SeenAGPRs;
        for (unsigned I = 0; I < Size / 4; ++I) {
          MachineOperand *Def = Defs[I].first;
          TargetInstrInfo::RegSubRegPair CopyToVGPR;
          if (Def->isImm() &&
              TII->isInlineConstant(*Def, AMDGPU::OPERAND_REG_INLINE_C_INT32)) {
            int64_t Imm = Def->getImm();

            auto Tmp = MRI->createVirtualRegister(&AMDGPU::AGPR_32RegClass);
            BuildMI(MBB, UseMI, DL,
                    TII->get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), Tmp).addImm(Imm);
            B.addReg(Tmp);
          } else if (Def->isReg() && TRI->isAGPR(*MRI, Def->getReg())) {
            auto Src = getRegSubRegPair(*Def);
            Def->setIsKill(false);
            if (!SeenAGPRs.insert(Src)) {
              // We cannot build a reg_sequence out of the same registers, they
              // must be copied. Better do it here before copyPhysReg() created
              // several reads to do the AGPR->VGPR->AGPR copy.
              CopyToVGPR = Src;
            } else {
              B.addReg(Src.Reg, Def->isUndef() ? RegState::Undef : 0,
                       Src.SubReg);
            }
          } else {
            assert(Def->isReg());
            Def->setIsKill(false);
            auto Src = getRegSubRegPair(*Def);

            // Direct copy from SGPR to AGPR is not possible. To avoid creation
            // of exploded copies SGPR->VGPR->AGPR in the copyPhysReg() later,
            // create a copy here and track if we already have such a copy.
            if (TRI->isSGPRReg(*MRI, Src.Reg)) {
              CopyToVGPR = Src;
            } else {
              auto Tmp = MRI->createVirtualRegister(&AMDGPU::AGPR_32RegClass);
              BuildMI(MBB, UseMI, DL, TII->get(AMDGPU::COPY), Tmp).add(*Def);
              B.addReg(Tmp);
            }
          }

          if (CopyToVGPR.Reg) {
            Register Vgpr;
            if (VGPRCopies.count(CopyToVGPR)) {
              Vgpr = VGPRCopies[CopyToVGPR];
            } else {
              Vgpr = MRI->createVirtualRegister(&AMDGPU::VGPR_32RegClass);
              BuildMI(MBB, UseMI, DL, TII->get(AMDGPU::COPY), Vgpr).add(*Def);
              VGPRCopies[CopyToVGPR] = Vgpr;
            }
            auto Tmp = MRI->createVirtualRegister(&AMDGPU::AGPR_32RegClass);
            BuildMI(MBB, UseMI, DL,
                    TII->get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), Tmp).addReg(Vgpr);
            B.addReg(Tmp);
          }

          B.addImm(Defs[I].second);
        }
        LLVM_DEBUG(dbgs() << "Folded " << *UseMI);
        return;
      }

      if (Size != 4)
        return;
      if (TRI->isAGPR(*MRI, UseMI->getOperand(0).getReg()) &&
          TRI->isVGPR(*MRI, UseMI->getOperand(1).getReg()))
        UseMI->setDesc(TII->get(AMDGPU::V_ACCVGPR_WRITE_B32_e64));
      else if (TRI->isVGPR(*MRI, UseMI->getOperand(0).getReg()) &&
               TRI->isAGPR(*MRI, UseMI->getOperand(1).getReg()))
        UseMI->setDesc(TII->get(AMDGPU::V_ACCVGPR_READ_B32_e64));
      else if (ST->hasGFX90AInsts() &&
               TRI->isAGPR(*MRI, UseMI->getOperand(0).getReg()) &&
               TRI->isAGPR(*MRI, UseMI->getOperand(1).getReg()))
        UseMI->setDesc(TII->get(AMDGPU::V_ACCVGPR_MOV_B32));
      return;
    }

    unsigned UseOpc = UseMI->getOpcode();
    if (UseOpc == AMDGPU::V_READFIRSTLANE_B32 ||
        (UseOpc == AMDGPU::V_READLANE_B32 &&
         (int)UseOpIdx ==
         AMDGPU::getNamedOperandIdx(UseOpc, AMDGPU::OpName::src0))) {
      // %vgpr = V_MOV_B32 imm
      // %sgpr = V_READFIRSTLANE_B32 %vgpr
      // =>
      // %sgpr = S_MOV_B32 imm
      if (FoldingImmLike) {
        if (execMayBeModifiedBeforeUse(*MRI,
                                       UseMI->getOperand(UseOpIdx).getReg(),
                                       *OpToFold.getParent(),
                                       *UseMI))
          return;

        UseMI->setDesc(TII->get(AMDGPU::S_MOV_B32));

        if (OpToFold.isImm())
          UseMI->getOperand(1).ChangeToImmediate(OpToFold.getImm());
        else
          UseMI->getOperand(1).ChangeToFrameIndex(OpToFold.getIndex());
        UseMI->removeOperand(2); // Remove exec read (or src1 for readlane)
        return;
      }

      if (OpToFold.isReg() && TRI->isSGPRReg(*MRI, OpToFold.getReg())) {
        if (execMayBeModifiedBeforeUse(*MRI,
                                       UseMI->getOperand(UseOpIdx).getReg(),
                                       *OpToFold.getParent(),
                                       *UseMI))
          return;

        // %vgpr = COPY %sgpr0
        // %sgpr1 = V_READFIRSTLANE_B32 %vgpr
        // =>
        // %sgpr1 = COPY %sgpr0
        UseMI->setDesc(TII->get(AMDGPU::COPY));
        UseMI->getOperand(1).setReg(OpToFold.getReg());
        UseMI->getOperand(1).setSubReg(OpToFold.getSubReg());
        UseMI->getOperand(1).setIsKill(false);
        UseMI->removeOperand(2); // Remove exec read (or src1 for readlane)
        return;
      }
    }

    const MCInstrDesc &UseDesc = UseMI->getDesc();

    // Don't fold into target independent nodes.  Target independent opcodes
    // don't have defined register classes.
    if (UseDesc.isVariadic() ||
        UseOp.isImplicit() ||
        UseDesc.OpInfo[UseOpIdx].RegClass == -1)
      return;
  }

  if (!FoldingImmLike) {
    if (OpToFold.isReg() && ST->needsAlignedVGPRs()) {
      // Don't fold if OpToFold doesn't hold an aligned register.
      const TargetRegisterClass *RC =
          TRI->getRegClassForReg(*MRI, OpToFold.getReg());
      if (TRI->hasVectorRegisters(RC) && OpToFold.getSubReg()) {
        unsigned SubReg = OpToFold.getSubReg();
        const TargetRegisterClass *SubRC = TRI->getSubRegClass(RC, SubReg);
        RC = TRI->getCompatibleSubRegClass(RC, SubRC, SubReg);
        if (RC)
          RC = SubRC;
      }

      if (!RC || !TRI->isProperlyAlignedRC(*RC))
        return;
    }

    tryAddToFoldList(FoldList, UseMI, UseOpIdx, &OpToFold, TII);

    // FIXME: We could try to change the instruction from 64-bit to 32-bit
    // to enable more folding opportunities.  The shrink operands pass
    // already does this.
    return;
  }


  const MCInstrDesc &FoldDesc = OpToFold.getParent()->getDesc();
  const TargetRegisterClass *FoldRC =
    TRI->getRegClass(FoldDesc.OpInfo[0].RegClass);

  // Split 64-bit constants into 32-bits for folding.
  if (UseOp.getSubReg() && AMDGPU::getRegBitWidth(FoldRC->getID()) == 64) {
    Register UseReg = UseOp.getReg();
    const TargetRegisterClass *UseRC = MRI->getRegClass(UseReg);

    if (AMDGPU::getRegBitWidth(UseRC->getID()) != 64)
      return;

    APInt Imm(64, OpToFold.getImm());
    if (UseOp.getSubReg() == AMDGPU::sub0) {
      Imm = Imm.getLoBits(32);
    } else {
      assert(UseOp.getSubReg() == AMDGPU::sub1);
      Imm = Imm.getHiBits(32);
    }

    MachineOperand ImmOp = MachineOperand::CreateImm(Imm.getSExtValue());
    tryAddToFoldList(FoldList, UseMI, UseOpIdx, &ImmOp, TII);
    return;
  }



  tryAddToFoldList(FoldList, UseMI, UseOpIdx, &OpToFold, TII);
}

static bool evalBinaryInstruction(unsigned Opcode, int32_t &Result,
                                  uint32_t LHS, uint32_t RHS) {
  switch (Opcode) {
  case AMDGPU::V_AND_B32_e64:
  case AMDGPU::V_AND_B32_e32:
  case AMDGPU::S_AND_B32:
    Result = LHS & RHS;
    return true;
  case AMDGPU::V_OR_B32_e64:
  case AMDGPU::V_OR_B32_e32:
  case AMDGPU::S_OR_B32:
    Result = LHS | RHS;
    return true;
  case AMDGPU::V_XOR_B32_e64:
  case AMDGPU::V_XOR_B32_e32:
  case AMDGPU::S_XOR_B32:
    Result = LHS ^ RHS;
    return true;
  case AMDGPU::S_XNOR_B32:
    Result = ~(LHS ^ RHS);
    return true;
  case AMDGPU::S_NAND_B32:
    Result = ~(LHS & RHS);
    return true;
  case AMDGPU::S_NOR_B32:
    Result = ~(LHS | RHS);
    return true;
  case AMDGPU::S_ANDN2_B32:
    Result = LHS & ~RHS;
    return true;
  case AMDGPU::S_ORN2_B32:
    Result = LHS | ~RHS;
    return true;
  case AMDGPU::V_LSHL_B32_e64:
  case AMDGPU::V_LSHL_B32_e32:
  case AMDGPU::S_LSHL_B32:
    // The instruction ignores the high bits for out of bounds shifts.
    Result = LHS << (RHS & 31);
    return true;
  case AMDGPU::V_LSHLREV_B32_e64:
  case AMDGPU::V_LSHLREV_B32_e32:
    Result = RHS << (LHS & 31);
    return true;
  case AMDGPU::V_LSHR_B32_e64:
  case AMDGPU::V_LSHR_B32_e32:
  case AMDGPU::S_LSHR_B32:
    Result = LHS >> (RHS & 31);
    return true;
  case AMDGPU::V_LSHRREV_B32_e64:
  case AMDGPU::V_LSHRREV_B32_e32:
    Result = RHS >> (LHS & 31);
    return true;
  case AMDGPU::V_ASHR_I32_e64:
  case AMDGPU::V_ASHR_I32_e32:
  case AMDGPU::S_ASHR_I32:
    Result = static_cast<int32_t>(LHS) >> (RHS & 31);
    return true;
  case AMDGPU::V_ASHRREV_I32_e64:
  case AMDGPU::V_ASHRREV_I32_e32:
    Result = static_cast<int32_t>(RHS) >> (LHS & 31);
    return true;
  default:
    return false;
  }
}

static unsigned getMovOpc(bool IsScalar) {
  return IsScalar ? AMDGPU::S_MOV_B32 : AMDGPU::V_MOV_B32_e32;
}

/// Remove any leftover implicit operands from mutating the instruction. e.g.
/// if we replace an s_and_b32 with a copy, we don't need the implicit scc def
/// anymore.
static void stripExtraCopyOperands(MachineInstr &MI) {
  const MCInstrDesc &Desc = MI.getDesc();
  unsigned NumOps = Desc.getNumOperands() +
                    Desc.getNumImplicitUses() +
                    Desc.getNumImplicitDefs();

  for (unsigned I = MI.getNumOperands() - 1; I >= NumOps; --I)
    MI.removeOperand(I);
}

static void mutateCopyOp(MachineInstr &MI, const MCInstrDesc &NewDesc) {
  MI.setDesc(NewDesc);
  stripExtraCopyOperands(MI);
}

static MachineOperand *getImmOrMaterializedImm(MachineRegisterInfo &MRI,
                                               MachineOperand &Op) {
  if (Op.isReg()) {
    // If this has a subregister, it obviously is a register source.
    if (Op.getSubReg() != AMDGPU::NoSubRegister || !Op.getReg().isVirtual())
      return &Op;

    MachineInstr *Def = MRI.getVRegDef(Op.getReg());
    if (Def && Def->isMoveImmediate()) {
      MachineOperand &ImmSrc = Def->getOperand(1);
      if (ImmSrc.isImm())
        return &ImmSrc;
    }
  }

  return &Op;
}

// Try to simplify operations with a constant that may appear after instruction
// selection.
// TODO: See if a frame index with a fixed offset can fold.
static bool tryConstantFoldOp(MachineRegisterInfo &MRI, const SIInstrInfo *TII,
                              MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();

  int Src0Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
  if (Src0Idx == -1)
    return false;
  MachineOperand *Src0 = getImmOrMaterializedImm(MRI, MI->getOperand(Src0Idx));

  if ((Opc == AMDGPU::V_NOT_B32_e64 || Opc == AMDGPU::V_NOT_B32_e32 ||
       Opc == AMDGPU::S_NOT_B32) &&
      Src0->isImm()) {
    MI->getOperand(1).ChangeToImmediate(~Src0->getImm());
    mutateCopyOp(*MI, TII->get(getMovOpc(Opc == AMDGPU::S_NOT_B32)));
    return true;
  }

  int Src1Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1);
  if (Src1Idx == -1)
    return false;
  MachineOperand *Src1 = getImmOrMaterializedImm(MRI, MI->getOperand(Src1Idx));

  if (!Src0->isImm() && !Src1->isImm())
    return false;

  // and k0, k1 -> v_mov_b32 (k0 & k1)
  // or k0, k1 -> v_mov_b32 (k0 | k1)
  // xor k0, k1 -> v_mov_b32 (k0 ^ k1)
  if (Src0->isImm() && Src1->isImm()) {
    int32_t NewImm;
    if (!evalBinaryInstruction(Opc, NewImm, Src0->getImm(), Src1->getImm()))
      return false;

    const SIRegisterInfo &TRI = TII->getRegisterInfo();
    bool IsSGPR = TRI.isSGPRReg(MRI, MI->getOperand(0).getReg());

    // Be careful to change the right operand, src0 may belong to a different
    // instruction.
    MI->getOperand(Src0Idx).ChangeToImmediate(NewImm);
    MI->removeOperand(Src1Idx);
    mutateCopyOp(*MI, TII->get(getMovOpc(IsSGPR)));
    return true;
  }

  if (!MI->isCommutable())
    return false;

  if (Src0->isImm() && !Src1->isImm()) {
    std::swap(Src0, Src1);
    std::swap(Src0Idx, Src1Idx);
  }

  int32_t Src1Val = static_cast<int32_t>(Src1->getImm());
  if (Opc == AMDGPU::V_OR_B32_e64 ||
      Opc == AMDGPU::V_OR_B32_e32 ||
      Opc == AMDGPU::S_OR_B32) {
    if (Src1Val == 0) {
      // y = or x, 0 => y = copy x
      MI->removeOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(AMDGPU::COPY));
    } else if (Src1Val == -1) {
      // y = or x, -1 => y = v_mov_b32 -1
      MI->removeOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(getMovOpc(Opc == AMDGPU::S_OR_B32)));
    } else
      return false;

    return true;
  }

  if (MI->getOpcode() == AMDGPU::V_AND_B32_e64 ||
      MI->getOpcode() == AMDGPU::V_AND_B32_e32 ||
      MI->getOpcode() == AMDGPU::S_AND_B32) {
    if (Src1Val == 0) {
      // y = and x, 0 => y = v_mov_b32 0
      MI->removeOperand(Src0Idx);
      mutateCopyOp(*MI, TII->get(getMovOpc(Opc == AMDGPU::S_AND_B32)));
    } else if (Src1Val == -1) {
      // y = and x, -1 => y = copy x
      MI->removeOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(AMDGPU::COPY));
      stripExtraCopyOperands(*MI);
    } else
      return false;

    return true;
  }

  if (MI->getOpcode() == AMDGPU::V_XOR_B32_e64 ||
      MI->getOpcode() == AMDGPU::V_XOR_B32_e32 ||
      MI->getOpcode() == AMDGPU::S_XOR_B32) {
    if (Src1Val == 0) {
      // y = xor x, 0 => y = copy x
      MI->removeOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(AMDGPU::COPY));
      return true;
    }
  }

  return false;
}

// Try to fold an instruction into a simpler one
bool SIFoldOperands::tryFoldCndMask(MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  if (Opc != AMDGPU::V_CNDMASK_B32_e32 && Opc != AMDGPU::V_CNDMASK_B32_e64 &&
      Opc != AMDGPU::V_CNDMASK_B64_PSEUDO)
    return false;

  MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
  MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
  if (!Src1->isIdenticalTo(*Src0)) {
    auto *Src0Imm = getImmOrMaterializedImm(*MRI, *Src0);
    auto *Src1Imm = getImmOrMaterializedImm(*MRI, *Src1);
    if (!Src1Imm->isIdenticalTo(*Src0Imm))
      return false;
  }

  int Src1ModIdx =
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1_modifiers);
  int Src0ModIdx =
      AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0_modifiers);
  if ((Src1ModIdx != -1 && MI.getOperand(Src1ModIdx).getImm() != 0) ||
      (Src0ModIdx != -1 && MI.getOperand(Src0ModIdx).getImm() != 0))
    return false;

  LLVM_DEBUG(dbgs() << "Folded " << MI << " into ");
  auto &NewDesc =
      TII->get(Src0->isReg() ? (unsigned)AMDGPU::COPY : getMovOpc(false));
  int Src2Idx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src2);
  if (Src2Idx != -1)
    MI.removeOperand(Src2Idx);
  MI.removeOperand(AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src1));
  if (Src1ModIdx != -1)
    MI.removeOperand(Src1ModIdx);
  if (Src0ModIdx != -1)
    MI.removeOperand(Src0ModIdx);
  mutateCopyOp(MI, NewDesc);
  LLVM_DEBUG(dbgs() << MI);
  return true;
}

bool SIFoldOperands::tryFoldZeroHighBits(MachineInstr &MI) const {
  if (MI.getOpcode() != AMDGPU::V_AND_B32_e64 &&
      MI.getOpcode() != AMDGPU::V_AND_B32_e32)
    return false;

  MachineOperand *Src0 = getImmOrMaterializedImm(*MRI, MI.getOperand(1));
  if (!Src0->isImm() || Src0->getImm() != 0xffff)
    return false;

  Register Src1 = MI.getOperand(2).getReg();
  MachineInstr *SrcDef = MRI->getVRegDef(Src1);
  if (ST->zeroesHigh16BitsOfDest(SrcDef->getOpcode())) {
    Register Dst = MI.getOperand(0).getReg();
    MRI->replaceRegWith(Dst, SrcDef->getOperand(0).getReg());
    MI.eraseFromParent();
    return true;
  }

  return false;
}

bool SIFoldOperands::foldInstOperand(MachineInstr &MI,
                                     MachineOperand &OpToFold) const {
  // We need mutate the operands of new mov instructions to add implicit
  // uses of EXEC, but adding them invalidates the use_iterator, so defer
  // this.
  SmallVector<MachineInstr *, 4> CopiesToReplace;
  SmallVector<FoldCandidate, 4> FoldList;
  MachineOperand &Dst = MI.getOperand(0);
  bool Changed = false;

  if (OpToFold.isImm()) {
    for (auto &UseMI :
         make_early_inc_range(MRI->use_nodbg_instructions(Dst.getReg()))) {
      // Folding the immediate may reveal operations that can be constant
      // folded or replaced with a copy. This can happen for example after
      // frame indices are lowered to constants or from splitting 64-bit
      // constants.
      //
      // We may also encounter cases where one or both operands are
      // immediates materialized into a register, which would ordinarily not
      // be folded due to multiple uses or operand constraints.
      if (tryConstantFoldOp(*MRI, TII, &UseMI)) {
        LLVM_DEBUG(dbgs() << "Constant folded " << UseMI);
        Changed = true;
      }
    }
  }

  SmallVector<MachineOperand *, 4> UsesToProcess;
  for (auto &Use : MRI->use_nodbg_operands(Dst.getReg()))
    UsesToProcess.push_back(&Use);
  for (auto U : UsesToProcess) {
    MachineInstr *UseMI = U->getParent();
    foldOperand(OpToFold, UseMI, UseMI->getOperandNo(U), FoldList,
                CopiesToReplace);
  }

  if (CopiesToReplace.empty() && FoldList.empty())
    return Changed;

  MachineFunction *MF = MI.getParent()->getParent();
  // Make sure we add EXEC uses to any new v_mov instructions created.
  for (MachineInstr *Copy : CopiesToReplace)
    Copy->addImplicitDefUseOperands(*MF);

  for (FoldCandidate &Fold : FoldList) {
    assert(!Fold.isReg() || Fold.OpToFold);
    if (Fold.isReg() && Fold.OpToFold->getReg().isVirtual()) {
      Register Reg = Fold.OpToFold->getReg();
      MachineInstr *DefMI = Fold.OpToFold->getParent();
      if (DefMI->readsRegister(AMDGPU::EXEC, TRI) &&
          execMayBeModifiedBeforeUse(*MRI, Reg, *DefMI, *Fold.UseMI))
        continue;
    }
    if (updateOperand(Fold, *TII, *TRI, *ST)) {
      // Clear kill flags.
      if (Fold.isReg()) {
        assert(Fold.OpToFold && Fold.OpToFold->isReg());
        // FIXME: Probably shouldn't bother trying to fold if not an
        // SGPR. PeepholeOptimizer can eliminate redundant VGPR->VGPR
        // copies.
        MRI->clearKillFlags(Fold.OpToFold->getReg());
      }
      LLVM_DEBUG(dbgs() << "Folded source from " << MI << " into OpNo "
                        << static_cast<int>(Fold.UseOpNo) << " of "
                        << *Fold.UseMI);
    } else if (Fold.isCommuted()) {
      // Restoring instruction's original operand order if fold has failed.
      TII->commuteInstruction(*Fold.UseMI, false);
    }
  }
  return true;
}

// Clamp patterns are canonically selected to v_max_* instructions, so only
// handle them.
const MachineOperand *SIFoldOperands::isClamp(const MachineInstr &MI) const {
  unsigned Op = MI.getOpcode();
  switch (Op) {
  case AMDGPU::V_MAX_F32_e64:
  case AMDGPU::V_MAX_F16_e64:
  case AMDGPU::V_MAX_F64_e64:
  case AMDGPU::V_PK_MAX_F16: {
    if (!TII->getNamedOperand(MI, AMDGPU::OpName::clamp)->getImm())
      return nullptr;

    // Make sure sources are identical.
    const MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
    if (!Src0->isReg() || !Src1->isReg() ||
        Src0->getReg() != Src1->getReg() ||
        Src0->getSubReg() != Src1->getSubReg() ||
        Src0->getSubReg() != AMDGPU::NoSubRegister)
      return nullptr;

    // Can't fold up if we have modifiers.
    if (TII->hasModifiersSet(MI, AMDGPU::OpName::omod))
      return nullptr;

    unsigned Src0Mods
      = TII->getNamedOperand(MI, AMDGPU::OpName::src0_modifiers)->getImm();
    unsigned Src1Mods
      = TII->getNamedOperand(MI, AMDGPU::OpName::src1_modifiers)->getImm();

    // Having a 0 op_sel_hi would require swizzling the output in the source
    // instruction, which we can't do.
    unsigned UnsetMods = (Op == AMDGPU::V_PK_MAX_F16) ? SISrcMods::OP_SEL_1
                                                      : 0u;
    if (Src0Mods != UnsetMods && Src1Mods != UnsetMods)
      return nullptr;
    return Src0;
  }
  default:
    return nullptr;
  }
}

// FIXME: Clamp for v_mad_mixhi_f16 handled during isel.
bool SIFoldOperands::tryFoldClamp(MachineInstr &MI) {
  const MachineOperand *ClampSrc = isClamp(MI);
  if (!ClampSrc || !MRI->hasOneNonDBGUser(ClampSrc->getReg()))
    return false;

  MachineInstr *Def = MRI->getVRegDef(ClampSrc->getReg());

  // The type of clamp must be compatible.
  if (TII->getClampMask(*Def) != TII->getClampMask(MI))
    return false;

  MachineOperand *DefClamp = TII->getNamedOperand(*Def, AMDGPU::OpName::clamp);
  if (!DefClamp)
    return false;

  LLVM_DEBUG(dbgs() << "Folding clamp " << *DefClamp << " into " << *Def);

  // Clamp is applied after omod, so it is OK if omod is set.
  DefClamp->setImm(1);
  MRI->replaceRegWith(MI.getOperand(0).getReg(), Def->getOperand(0).getReg());
  MI.eraseFromParent();

  // Use of output modifiers forces VOP3 encoding for a VOP2 mac/fmac
  // instruction, so we might as well convert it to the more flexible VOP3-only
  // mad/fma form.
  if (TII->convertToThreeAddress(*Def, nullptr, nullptr))
    Def->eraseFromParent();

  return true;
}

static int getOModValue(unsigned Opc, int64_t Val) {
  switch (Opc) {
  case AMDGPU::V_MUL_F64_e64: {
    switch (Val) {
    case 0x3fe0000000000000: // 0.5
      return SIOutMods::DIV2;
    case 0x4000000000000000: // 2.0
      return SIOutMods::MUL2;
    case 0x4010000000000000: // 4.0
      return SIOutMods::MUL4;
    default:
      return SIOutMods::NONE;
    }
  }
  case AMDGPU::V_MUL_F32_e64: {
    switch (static_cast<uint32_t>(Val)) {
    case 0x3f000000: // 0.5
      return SIOutMods::DIV2;
    case 0x40000000: // 2.0
      return SIOutMods::MUL2;
    case 0x40800000: // 4.0
      return SIOutMods::MUL4;
    default:
      return SIOutMods::NONE;
    }
  }
  case AMDGPU::V_MUL_F16_e64: {
    switch (static_cast<uint16_t>(Val)) {
    case 0x3800: // 0.5
      return SIOutMods::DIV2;
    case 0x4000: // 2.0
      return SIOutMods::MUL2;
    case 0x4400: // 4.0
      return SIOutMods::MUL4;
    default:
      return SIOutMods::NONE;
    }
  }
  default:
    llvm_unreachable("invalid mul opcode");
  }
}

// FIXME: Does this really not support denormals with f16?
// FIXME: Does this need to check IEEE mode bit? SNaNs are generally not
// handled, so will anything other than that break?
std::pair<const MachineOperand *, int>
SIFoldOperands::isOMod(const MachineInstr &MI) const {
  unsigned Op = MI.getOpcode();
  switch (Op) {
  case AMDGPU::V_MUL_F64_e64:
  case AMDGPU::V_MUL_F32_e64:
  case AMDGPU::V_MUL_F16_e64: {
    // If output denormals are enabled, omod is ignored.
    if ((Op == AMDGPU::V_MUL_F32_e64 && MFI->getMode().FP32OutputDenormals) ||
        ((Op == AMDGPU::V_MUL_F64_e64 || Op == AMDGPU::V_MUL_F16_e64) &&
         MFI->getMode().FP64FP16OutputDenormals))
      return std::make_pair(nullptr, SIOutMods::NONE);

    const MachineOperand *RegOp = nullptr;
    const MachineOperand *ImmOp = nullptr;
    const MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);
    if (Src0->isImm()) {
      ImmOp = Src0;
      RegOp = Src1;
    } else if (Src1->isImm()) {
      ImmOp = Src1;
      RegOp = Src0;
    } else
      return std::make_pair(nullptr, SIOutMods::NONE);

    int OMod = getOModValue(Op, ImmOp->getImm());
    if (OMod == SIOutMods::NONE ||
        TII->hasModifiersSet(MI, AMDGPU::OpName::src0_modifiers) ||
        TII->hasModifiersSet(MI, AMDGPU::OpName::src1_modifiers) ||
        TII->hasModifiersSet(MI, AMDGPU::OpName::omod) ||
        TII->hasModifiersSet(MI, AMDGPU::OpName::clamp))
      return std::make_pair(nullptr, SIOutMods::NONE);

    return std::make_pair(RegOp, OMod);
  }
  case AMDGPU::V_ADD_F64_e64:
  case AMDGPU::V_ADD_F32_e64:
  case AMDGPU::V_ADD_F16_e64: {
    // If output denormals are enabled, omod is ignored.
    if ((Op == AMDGPU::V_ADD_F32_e64 && MFI->getMode().FP32OutputDenormals) ||
        ((Op == AMDGPU::V_ADD_F64_e64 || Op == AMDGPU::V_ADD_F16_e64) &&
         MFI->getMode().FP64FP16OutputDenormals))
      return std::make_pair(nullptr, SIOutMods::NONE);

    // Look through the DAGCombiner canonicalization fmul x, 2 -> fadd x, x
    const MachineOperand *Src0 = TII->getNamedOperand(MI, AMDGPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(MI, AMDGPU::OpName::src1);

    if (Src0->isReg() && Src1->isReg() && Src0->getReg() == Src1->getReg() &&
        Src0->getSubReg() == Src1->getSubReg() &&
        !TII->hasModifiersSet(MI, AMDGPU::OpName::src0_modifiers) &&
        !TII->hasModifiersSet(MI, AMDGPU::OpName::src1_modifiers) &&
        !TII->hasModifiersSet(MI, AMDGPU::OpName::clamp) &&
        !TII->hasModifiersSet(MI, AMDGPU::OpName::omod))
      return std::make_pair(Src0, SIOutMods::MUL2);

    return std::make_pair(nullptr, SIOutMods::NONE);
  }
  default:
    return std::make_pair(nullptr, SIOutMods::NONE);
  }
}

// FIXME: Does this need to check IEEE bit on function?
bool SIFoldOperands::tryFoldOMod(MachineInstr &MI) {
  const MachineOperand *RegOp;
  int OMod;
  std::tie(RegOp, OMod) = isOMod(MI);
  if (OMod == SIOutMods::NONE || !RegOp->isReg() ||
      RegOp->getSubReg() != AMDGPU::NoSubRegister ||
      !MRI->hasOneNonDBGUser(RegOp->getReg()))
    return false;

  MachineInstr *Def = MRI->getVRegDef(RegOp->getReg());
  MachineOperand *DefOMod = TII->getNamedOperand(*Def, AMDGPU::OpName::omod);
  if (!DefOMod || DefOMod->getImm() != SIOutMods::NONE)
    return false;

  // Clamp is applied after omod. If the source already has clamp set, don't
  // fold it.
  if (TII->hasModifiersSet(*Def, AMDGPU::OpName::clamp))
    return false;

  LLVM_DEBUG(dbgs() << "Folding omod " << MI << " into " << *Def);

  DefOMod->setImm(OMod);
  MRI->replaceRegWith(MI.getOperand(0).getReg(), Def->getOperand(0).getReg());
  MI.eraseFromParent();

  // Use of output modifiers forces VOP3 encoding for a VOP2 mac/fmac
  // instruction, so we might as well convert it to the more flexible VOP3-only
  // mad/fma form.
  if (TII->convertToThreeAddress(*Def, nullptr, nullptr))
    Def->eraseFromParent();

  return true;
}

// Try to fold a reg_sequence with vgpr output and agpr inputs into an
// instruction which can take an agpr. So far that means a store.
bool SIFoldOperands::tryFoldRegSequence(MachineInstr &MI) {
  assert(MI.isRegSequence());
  auto Reg = MI.getOperand(0).getReg();

  if (!ST->hasGFX90AInsts() || !TRI->isVGPR(*MRI, Reg) ||
      !MRI->hasOneNonDBGUse(Reg))
    return false;

  SmallVector<std::pair<MachineOperand*, unsigned>, 32> Defs;
  if (!getRegSeqInit(Defs, Reg, MCOI::OPERAND_REGISTER, TII, *MRI))
    return false;

  for (auto &Def : Defs) {
    const auto *Op = Def.first;
    if (!Op->isReg())
      return false;
    if (TRI->isAGPR(*MRI, Op->getReg()))
      continue;
    // Maybe this is a COPY from AREG
    const MachineInstr *SubDef = MRI->getVRegDef(Op->getReg());
    if (!SubDef || !SubDef->isCopy() || SubDef->getOperand(1).getSubReg())
      return false;
    if (!TRI->isAGPR(*MRI, SubDef->getOperand(1).getReg()))
      return false;
  }

  MachineOperand *Op = &*MRI->use_nodbg_begin(Reg);
  MachineInstr *UseMI = Op->getParent();
  while (UseMI->isCopy() && !Op->getSubReg()) {
    Reg = UseMI->getOperand(0).getReg();
    if (!TRI->isVGPR(*MRI, Reg) || !MRI->hasOneNonDBGUse(Reg))
      return false;
    Op = &*MRI->use_nodbg_begin(Reg);
    UseMI = Op->getParent();
  }

  if (Op->getSubReg())
    return false;

  unsigned OpIdx = Op - &UseMI->getOperand(0);
  const MCInstrDesc &InstDesc = UseMI->getDesc();
  const TargetRegisterClass *OpRC =
      TII->getRegClass(InstDesc, OpIdx, TRI, *MI.getMF());
  if (!OpRC || !TRI->isVectorSuperClass(OpRC))
    return false;

  const auto *NewDstRC = TRI->getEquivalentAGPRClass(MRI->getRegClass(Reg));
  auto Dst = MRI->createVirtualRegister(NewDstRC);
  auto RS = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                    TII->get(AMDGPU::REG_SEQUENCE), Dst);

  for (unsigned I = 0; I < Defs.size(); ++I) {
    MachineOperand *Def = Defs[I].first;
    Def->setIsKill(false);
    if (TRI->isAGPR(*MRI, Def->getReg())) {
      RS.add(*Def);
    } else { // This is a copy
      MachineInstr *SubDef = MRI->getVRegDef(Def->getReg());
      SubDef->getOperand(1).setIsKill(false);
      RS.addReg(SubDef->getOperand(1).getReg(), 0, Def->getSubReg());
    }
    RS.addImm(Defs[I].second);
  }

  Op->setReg(Dst);
  if (!TII->isOperandLegal(*UseMI, OpIdx, Op)) {
    Op->setReg(Reg);
    RS->eraseFromParent();
    return false;
  }

  LLVM_DEBUG(dbgs() << "Folded " << *RS << " into " << *UseMI);

  // Erase the REG_SEQUENCE eagerly, unless we followed a chain of COPY users,
  // in which case we can erase them all later in runOnMachineFunction.
  if (MRI->use_nodbg_empty(MI.getOperand(0).getReg()))
    MI.eraseFromParent();
  return true;
}

// Try to hoist an AGPR to VGPR copy out of the loop across a LCSSA PHI.
// This should allow folding of an AGPR into a consumer which may support it.
// I.e.:
//
// loop:                             // loop:
//   %1:vreg = COPY %0:areg          // exit:
// exit:                          => //   %1:areg = PHI %0:areg, %loop
//   %2:vreg = PHI %1:vreg, %loop    //   %2:vreg = COPY %1:areg
bool SIFoldOperands::tryFoldLCSSAPhi(MachineInstr &PHI) {
  assert(PHI.isPHI());

  if (PHI.getNumExplicitOperands() != 3) // Single input LCSSA PHI
    return false;

  Register PhiIn = PHI.getOperand(1).getReg();
  Register PhiOut = PHI.getOperand(0).getReg();
  if (PHI.getOperand(1).getSubReg() ||
      !TRI->isVGPR(*MRI, PhiIn) || !TRI->isVGPR(*MRI, PhiOut))
    return false;

  // A single use should not matter for correctness, but if it has another use
  // inside the loop we may perform copy twice in a worst case.
  if (!MRI->hasOneNonDBGUse(PhiIn))
    return false;

  MachineInstr *Copy = MRI->getVRegDef(PhiIn);
  if (!Copy || !Copy->isCopy())
    return false;

  Register CopyIn = Copy->getOperand(1).getReg();
  if (!TRI->isAGPR(*MRI, CopyIn) || Copy->getOperand(1).getSubReg())
    return false;

  const TargetRegisterClass *ARC = MRI->getRegClass(CopyIn);
  Register NewReg = MRI->createVirtualRegister(ARC);
  PHI.getOperand(1).setReg(CopyIn);
  PHI.getOperand(0).setReg(NewReg);

  MachineBasicBlock *MBB = PHI.getParent();
  BuildMI(*MBB, MBB->getFirstNonPHI(), Copy->getDebugLoc(),
          TII->get(AMDGPU::COPY), PhiOut)
    .addReg(NewReg, RegState::Kill);
  Copy->eraseFromParent(); // We know this copy had a single use.

  LLVM_DEBUG(dbgs() << "Folded " << PHI);

  return true;
}

// Attempt to convert VGPR load to an AGPR load.
bool SIFoldOperands::tryFoldLoad(MachineInstr &MI) {
  assert(MI.mayLoad());
  if (!ST->hasGFX90AInsts() || MI.getNumExplicitDefs() != 1)
    return false;

  MachineOperand &Def = MI.getOperand(0);
  if (!Def.isDef())
    return false;

  Register DefReg = Def.getReg();

  if (DefReg.isPhysical() || !TRI->isVGPR(*MRI, DefReg))
    return false;

  SmallVector<const MachineInstr*, 8> Users;
  SmallVector<Register, 8> MoveRegs;
  for (const MachineInstr &I : MRI->use_nodbg_instructions(DefReg)) {
    Users.push_back(&I);
  }
  if (Users.empty())
    return false;

  // Check that all uses a copy to an agpr or a reg_sequence producing an agpr.
  while (!Users.empty()) {
    const MachineInstr *I = Users.pop_back_val();
    if (!I->isCopy() && !I->isRegSequence())
      return false;
    Register DstReg = I->getOperand(0).getReg();
    if (TRI->isAGPR(*MRI, DstReg))
      continue;
    MoveRegs.push_back(DstReg);
    for (const MachineInstr &U : MRI->use_nodbg_instructions(DstReg)) {
      Users.push_back(&U);
    }
  }

  const TargetRegisterClass *RC = MRI->getRegClass(DefReg);
  MRI->setRegClass(DefReg, TRI->getEquivalentAGPRClass(RC));
  if (!TII->isOperandLegal(MI, 0, &Def)) {
    MRI->setRegClass(DefReg, RC);
    return false;
  }

  while (!MoveRegs.empty()) {
    Register Reg = MoveRegs.pop_back_val();
    MRI->setRegClass(Reg, TRI->getEquivalentAGPRClass(MRI->getRegClass(Reg)));
  }

  LLVM_DEBUG(dbgs() << "Folded " << MI);

  return true;
}

bool SIFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();

  // omod is ignored by hardware if IEEE bit is enabled. omod also does not
  // correctly handle signed zeros.
  //
  // FIXME: Also need to check strictfp
  bool IsIEEEMode = MFI->getMode().IEEE;
  bool HasNSZ = MFI->hasNoSignedZerosFPMath();

  bool Changed = false;
  for (MachineBasicBlock *MBB : depth_first(&MF)) {
    MachineOperand *CurrentKnownM0Val = nullptr;
    for (auto &MI : make_early_inc_range(*MBB)) {
      Changed |= tryFoldCndMask(MI);

      if (tryFoldZeroHighBits(MI)) {
        Changed = true;
        continue;
      }

      if (MI.isRegSequence() && tryFoldRegSequence(MI)) {
        Changed = true;
        continue;
      }

      if (MI.isPHI() && tryFoldLCSSAPhi(MI)) {
        Changed = true;
        continue;
      }

      if (MI.mayLoad() && tryFoldLoad(MI)) {
        Changed = true;
        continue;
      }

      if (!TII->isFoldableCopy(MI)) {
        // Saw an unknown clobber of m0, so we no longer know what it is.
        if (CurrentKnownM0Val && MI.modifiesRegister(AMDGPU::M0, TRI))
          CurrentKnownM0Val = nullptr;

        // TODO: Omod might be OK if there is NSZ only on the source
        // instruction, and not the omod multiply.
        if (IsIEEEMode || (!HasNSZ && !MI.getFlag(MachineInstr::FmNsz)) ||
            !tryFoldOMod(MI))
          Changed |= tryFoldClamp(MI);

        continue;
      }

      // Specially track simple redefs of m0 to the same value in a block, so we
      // can erase the later ones.
      if (MI.getOperand(0).getReg() == AMDGPU::M0) {
        MachineOperand &NewM0Val = MI.getOperand(1);
        if (CurrentKnownM0Val && CurrentKnownM0Val->isIdenticalTo(NewM0Val)) {
          MI.eraseFromParent();
          Changed = true;
          continue;
        }

        // We aren't tracking other physical registers
        CurrentKnownM0Val = (NewM0Val.isReg() && NewM0Val.getReg().isPhysical()) ?
          nullptr : &NewM0Val;
        continue;
      }

      MachineOperand &OpToFold = MI.getOperand(1);
      bool FoldingImm =
          OpToFold.isImm() || OpToFold.isFI() || OpToFold.isGlobal();

      // FIXME: We could also be folding things like TargetIndexes.
      if (!FoldingImm && !OpToFold.isReg())
        continue;

      if (OpToFold.isReg() && !OpToFold.getReg().isVirtual())
        continue;

      // Prevent folding operands backwards in the function. For example,
      // the COPY opcode must not be replaced by 1 in this example:
      //
      //    %3 = COPY %vgpr0; VGPR_32:%3
      //    ...
      //    %vgpr0 = V_MOV_B32_e32 1, implicit %exec
      if (!MI.getOperand(0).getReg().isVirtual())
        continue;

      Changed |= foldInstOperand(MI, OpToFold);

      // If we managed to fold all uses of this copy then we might as well
      // delete it now.
      // The only reason we need to follow chains of copies here is that
      // tryFoldRegSequence looks forward through copies before folding a
      // REG_SEQUENCE into its eventual users.
      auto *InstToErase = &MI;
      while (MRI->use_nodbg_empty(InstToErase->getOperand(0).getReg())) {
        auto &SrcOp = InstToErase->getOperand(1);
        auto SrcReg = SrcOp.isReg() ? SrcOp.getReg() : Register();
        InstToErase->eraseFromParent();
        Changed = true;
        InstToErase = nullptr;
        if (!SrcReg || SrcReg.isPhysical())
          break;
        InstToErase = MRI->getVRegDef(SrcReg);
        if (!InstToErase || !TII->isFoldableCopy(*InstToErase))
          break;
      }
      if (InstToErase && InstToErase->isRegSequence() &&
          MRI->use_nodbg_empty(InstToErase->getOperand(0).getReg())) {
        InstToErase->eraseFromParent();
        Changed = true;
      }
    }
  }
  return Changed;
}
