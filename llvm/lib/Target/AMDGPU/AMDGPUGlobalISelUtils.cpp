//===- AMDGPUGlobalISelUtils.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPURegisterBankInfo.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

using namespace llvm;
using namespace AMDGPU;
using namespace MIPatternMatch;

std::pair<Register, unsigned>
AMDGPU::getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg,
                                  GISelKnownBits *KnownBits, bool CheckNUW) {
  MachineInstr *Def = getDefIgnoringCopies(Reg, MRI);
  if (Def->getOpcode() == TargetOpcode::G_CONSTANT) {
    unsigned Offset;
    const MachineOperand &Op = Def->getOperand(1);
    if (Op.isImm())
      Offset = Op.getImm();
    else
      Offset = Op.getCImm()->getZExtValue();

    return std::pair(Register(), Offset);
  }

  int64_t Offset;
  if (Def->getOpcode() == TargetOpcode::G_ADD) {
    // A 32-bit (address + offset) should not cause unsigned 32-bit integer
    // wraparound, because s_load instructions perform the addition in 64 bits.
    if (CheckNUW && !Def->getFlag(MachineInstr::NoUWrap)) {
      assert(MRI.getType(Reg).getScalarSizeInBits() == 32);
      return std::pair(Reg, 0);
    }
    // TODO: Handle G_OR used for add case
    if (mi_match(Def->getOperand(2).getReg(), MRI, m_ICst(Offset)))
      return std::pair(Def->getOperand(1).getReg(), Offset);

    // FIXME: matcher should ignore copies
    if (mi_match(Def->getOperand(2).getReg(), MRI, m_Copy(m_ICst(Offset))))
      return std::pair(Def->getOperand(1).getReg(), Offset);
  }

  Register Base;
  if (KnownBits && mi_match(Reg, MRI, m_GOr(m_Reg(Base), m_ICst(Offset))) &&
      KnownBits->maskedValueIsZero(Base, APInt(32, Offset)))
    return std::pair(Base, Offset);

  // Handle G_PTRTOINT (G_PTR_ADD base, const) case
  if (Def->getOpcode() == TargetOpcode::G_PTRTOINT) {
    MachineInstr *Base;
    if (mi_match(Def->getOperand(1).getReg(), MRI,
                 m_GPtrAdd(m_MInstr(Base), m_ICst(Offset)))) {
      // If Base was int converted to pointer, simply return int and offset.
      if (Base->getOpcode() == TargetOpcode::G_INTTOPTR)
        return std::pair(Base->getOperand(1).getReg(), Offset);

      // Register returned here will be of pointer type.
      return std::pair(Base->getOperand(0).getReg(), Offset);
    }
  }

  return std::pair(Reg, 0);
}

IntrinsicLaneMaskAnalyzer::IntrinsicLaneMaskAnalyzer(MachineFunction &MF)
    : MRI(MF.getRegInfo()) {
  initLaneMaskIntrinsics(MF);
}

bool IntrinsicLaneMaskAnalyzer::isS32S64LaneMask(Register Reg) {
  return S32S64LaneMask.contains(Reg);
}

void IntrinsicLaneMaskAnalyzer::initLaneMaskIntrinsics(MachineFunction &MF) {
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      if (MI.getOpcode() == AMDGPU::G_INTRINSIC &&
          MI.getOperand(MI.getNumExplicitDefs()).getIntrinsicID() ==
              Intrinsic::amdgcn_if_break) {
        S32S64LaneMask.insert(MI.getOperand(3).getReg());
        findLCSSAPhi(MI.getOperand(0).getReg());
      }

      if (MI.getOpcode() == AMDGPU::SI_IF ||
          MI.getOpcode() == AMDGPU::SI_ELSE) {
        findLCSSAPhi(MI.getOperand(0).getReg());
      }
    }
  }
}

void IntrinsicLaneMaskAnalyzer::findLCSSAPhi(Register Reg) {
  S32S64LaneMask.insert(Reg);
  for (auto &LCSSAPhi : MRI.use_instructions(Reg)) {
    if (LCSSAPhi.isPHI())
      S32S64LaneMask.insert(LCSSAPhi.getOperand(0).getReg());
  }
}

MachineInstrBuilder AMDGPU::buildReadAnyLaneB32(MachineIRBuilder &B,
                                                const DstOp &SgprDst,
                                                const SrcOp &VgprSrc,
                                                const RegisterBankInfo &RBI) {
  auto RFL = B.buildInstr(AMDGPU::G_READANYLANE, {SgprDst}, {VgprSrc});
  Register Dst = RFL->getOperand(0).getReg();
  Register Src = RFL->getOperand(1).getReg();
  MachineRegisterInfo &MRI = *B.getMRI();
  if (!MRI.getRegBankOrNull(Dst))
    MRI.setRegBank(Dst, RBI.getRegBank(SGPRRegBankID));
  if (!MRI.getRegBankOrNull(Src))
    MRI.setRegBank(Src, RBI.getRegBank(VGPRRegBankID));
  return RFL;
}

MachineInstrBuilder
AMDGPU::buildReadAnyLaneSequenceOfB32(MachineIRBuilder &B, const DstOp &SgprDst,
                                      const SrcOp &VgprSrc, LLT B32Ty,
                                      const RegisterBankInfo &RBI) {
  MachineRegisterInfo &MRI = *B.getMRI();
  SmallVector<Register, 8> SgprDstParts;
  auto Unmerge = B.buildUnmerge(B32Ty, VgprSrc);
  for (unsigned i = 0; i < Unmerge->getNumOperands() - 1; ++i) {
    SgprDstParts.push_back(
        buildReadAnyLaneB32(B, B32Ty, Unmerge.getReg(i), RBI).getReg(0));
  }

  auto Merge = B.buildMergeLikeInstr(SgprDst, SgprDstParts);
  MRI.setRegBank(Merge.getReg(0), RBI.getRegBank(AMDGPU::SGPRRegBankID));
  return Merge;
}

MachineInstrBuilder
AMDGPU::buildReadAnyLaneSequenceOfS64(MachineIRBuilder &B, const DstOp &SgprDst,
                                      const SrcOp &VgprSrc,
                                      const RegisterBankInfo &RBI) {
  LLT S32 = LLT::scalar(32);
  LLT S64 = LLT::scalar(64);
  MachineRegisterInfo &MRI = *B.getMRI();
  SmallVector<Register, 8> SgprDstParts;
  auto Unmerge = B.buildUnmerge(S64, VgprSrc);

  for (unsigned i = 0; i < Unmerge->getNumOperands() - 1; ++i) {
    MRI.setRegBank(Unmerge.getReg(i), RBI.getRegBank(AMDGPU::VGPRRegBankID));
    auto Unmerge64 = B.buildUnmerge(S32, Unmerge.getReg(i));
    SmallVector<Register, 2> Unmerge64Parts;
    Unmerge64Parts.push_back(
        buildReadAnyLaneB32(B, S32, Unmerge64.getReg(0), RBI).getReg(0));
    Unmerge64Parts.push_back(
        buildReadAnyLaneB32(B, S32, Unmerge64.getReg(1), RBI).getReg(0));
    Register MergeReg = B.buildMergeLikeInstr(S64, Unmerge64Parts).getReg(0);
    MRI.setRegBank(MergeReg, RBI.getRegBank(AMDGPU::SGPRRegBankID));
    SgprDstParts.push_back(MergeReg);
  }

  auto Merge = B.buildMergeLikeInstr(SgprDst, SgprDstParts);
  MRI.setRegBank(Merge.getReg(0), RBI.getRegBank(AMDGPU::SGPRRegBankID));
  return Merge;
}

MachineInstrBuilder AMDGPU::buildReadAnyLane(MachineIRBuilder &B,
                                             const DstOp &SgprDst,
                                             const SrcOp &VgprSrc,
                                             const RegisterBankInfo &RBI) {
  MachineRegisterInfo &MRI = *B.getMRI();
  LLT S16 = LLT::scalar(16);
  LLT S32 = LLT::scalar(32);
  LLT S64 = LLT::scalar(64);
  LLT S256 = LLT::scalar(256);
  LLT V2S16 = LLT::fixed_vector(2, 16);
  LLT Ty = SgprDst.getLLTTy(MRI);

  if (Ty == S16) {
    return B.buildTrunc(
        SgprDst, buildReadAnyLaneB32(B, S32, B.buildAnyExt(S32, VgprSrc), RBI));
  }

  if (Ty == S32 || Ty == V2S16 ||
      (Ty.isPointer() && Ty.getSizeInBits() == 32)) {
    return buildReadAnyLaneB32(B, SgprDst, VgprSrc, RBI);
  }

  if (Ty == S64 || Ty == S256 || (Ty.isPointer() && Ty.getSizeInBits() == 64) ||
      (Ty.isVector() && Ty.getElementType() == S32)) {
    return buildReadAnyLaneSequenceOfB32(B, SgprDst, VgprSrc, S32, RBI);
  }

  if (Ty.isVector() && Ty.getElementType() == S16) {
    return buildReadAnyLaneSequenceOfB32(B, SgprDst, VgprSrc, V2S16, RBI);
  }

  if (Ty.isVector() && Ty.getElementType() == S64) {
    return buildReadAnyLaneSequenceOfS64(B, SgprDst, VgprSrc, RBI);
  }

  llvm_unreachable("Type not supported");
}

void AMDGPU::buildReadAnyLaneDst(MachineIRBuilder &B, MachineInstr &MI,
                                 const RegisterBankInfo &RBI) {
  MachineRegisterInfo &MRI = *B.getMRI();
  Register Dst = MI.getOperand(0).getReg();
  const RegisterBank *DstBank = MRI.getRegBankOrNull(Dst);
  if (DstBank != &RBI.getRegBank(AMDGPU::SGPRRegBankID))
    return;

  Register VgprDst = MRI.createGenericVirtualRegister(MRI.getType(Dst));
  MRI.setRegBank(VgprDst, RBI.getRegBank(AMDGPU::VGPRRegBankID));

  MI.getOperand(0).setReg(VgprDst);
  MachineBasicBlock *MBB = MI.getParent();
  B.setInsertPt(*MBB, std::next(MI.getIterator()));
  // readAnyLane VgprDst into Dst after MI.
  buildReadAnyLane(B, Dst, VgprDst, RBI);
  return;
}

bool AMDGPU::isLaneMask(Register Reg, MachineRegisterInfo &MRI,
                        const SIRegisterInfo *TRI) {
  const RegisterBank *RB = MRI.getRegBankOrNull(Reg);
  if (RB && RB->getID() == VCCRegBankID)
    return true;

  const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
  if (RC && TRI->isSGPRClass(RC) && MRI.getType(Reg) == LLT::scalar(1))
    return true;

  return false;
}

bool AMDGPU::isSgprRB(Register Reg, MachineRegisterInfo &MRI) {
  const RegisterBank *RB = MRI.getRegBankOrNull(Reg);
  if (RB && RB->getID() == SGPRRegBankID)
    return true;

  return false;
}

bool AMDGPU::isVgprRB(Register Reg, MachineRegisterInfo &MRI) {
  const RegisterBank *RB = MRI.getRegBankOrNull(Reg);
  if (RB && RB->getID() == VGPRRegBankID)
    return true;

  return false;
}

void AMDGPU::cleanUpAfterCombine(MachineInstr &MI, MachineRegisterInfo &MRI,
                                 MachineInstr *Optional0) {
  MI.eraseFromParent();
  if (Optional0 && isTriviallyDead(*Optional0, MRI))
    Optional0->eraseFromParent();
}

bool AMDGPU::hasSGPRS1(MachineFunction &MF, MachineRegisterInfo &MRI) {
  for (auto &MBB : MF) {
    for (auto &MI : make_early_inc_range(MBB)) {
      for (MachineOperand &Op : MI.operands()) {
        if (!Op.isReg())
          continue;

        Register Reg = Op.getReg();
        if (!Reg.isVirtual())
          continue;

        if (!isSgprRB(Reg, MRI) || MRI.getType(Reg) != LLT::scalar(1))
          continue;

        MI.getParent()->dump();
        MI.dump();
        return true;
      }
    }
  }
  return false;
}

bool AMDGPU::isS1(Register Reg, MachineRegisterInfo &MRI) {
  return MRI.getType(Reg) == LLT::scalar(1);
}
