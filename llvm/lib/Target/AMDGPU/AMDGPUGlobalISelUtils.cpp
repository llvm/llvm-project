//===- AMDGPUGlobalISelUtils.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUGlobalISelUtils.h"
#include "AMDGPURegisterBankInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
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
      KnownBits->maskedValueIsZero(Base, APInt(32, Offset, /*isSigned=*/true)))
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

bool IntrinsicLaneMaskAnalyzer::isS32S64LaneMask(Register Reg) const {
  return S32S64LaneMask.contains(Reg);
}

void IntrinsicLaneMaskAnalyzer::initLaneMaskIntrinsics(MachineFunction &MF) {
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      GIntrinsic *GI = dyn_cast<GIntrinsic>(&MI);
      if (GI && GI->is(Intrinsic::amdgcn_if_break)) {
        S32S64LaneMask.insert(MI.getOperand(3).getReg());
        S32S64LaneMask.insert(MI.getOperand(0).getReg());
      }

      if (MI.getOpcode() == AMDGPU::SI_IF ||
          MI.getOpcode() == AMDGPU::SI_ELSE) {
        S32S64LaneMask.insert(MI.getOperand(0).getReg());
      }
    }
  }
}

static LLT getReadAnyLaneSplitTy(LLT Ty) {
  if (Ty.isVector()) {
    LLT ElTy = Ty.getElementType();
    if (ElTy.getSizeInBits() == 16)
      return LLT::fixed_vector(2, ElTy);
    // S32, S64 or pointer
    return ElTy;
  }

  // Large scalars and 64-bit pointers
  return LLT::scalar(32);
}

static Register buildReadAnyLane(MachineIRBuilder &B, Register VgprSrc,
                                 const RegisterBankInfo &RBI);

static void unmergeReadAnyLane(MachineIRBuilder &B,
                               SmallVectorImpl<Register> &SgprDstParts,
                               LLT UnmergeTy, Register VgprSrc,
                               const RegisterBankInfo &RBI) {
  const RegisterBank *VgprRB = &RBI.getRegBank(AMDGPU::VGPRRegBankID);
  auto Unmerge = B.buildUnmerge({VgprRB, UnmergeTy}, VgprSrc);
  for (unsigned i = 0; i < Unmerge->getNumOperands() - 1; ++i) {
    SgprDstParts.push_back(buildReadAnyLane(B, Unmerge.getReg(i), RBI));
  }
}

static Register buildReadAnyLane(MachineIRBuilder &B, Register VgprSrc,
                                 const RegisterBankInfo &RBI) {
  LLT Ty = B.getMRI()->getType(VgprSrc);
  const RegisterBank *SgprRB = &RBI.getRegBank(AMDGPU::SGPRRegBankID);
  if (Ty.getSizeInBits() == 32) {
    return B.buildInstr(AMDGPU::G_AMDGPU_READANYLANE, {{SgprRB, Ty}}, {VgprSrc})
        .getReg(0);
  }

  SmallVector<Register, 8> SgprDstParts;
  unmergeReadAnyLane(B, SgprDstParts, getReadAnyLaneSplitTy(Ty), VgprSrc, RBI);

  return B.buildMergeLikeInstr({SgprRB, Ty}, SgprDstParts).getReg(0);
}

void AMDGPU::buildReadAnyLane(MachineIRBuilder &B, Register SgprDst,
                              Register VgprSrc, const RegisterBankInfo &RBI) {
  LLT Ty = B.getMRI()->getType(VgprSrc);
  if (Ty.getSizeInBits() == 32) {
    B.buildInstr(AMDGPU::G_AMDGPU_READANYLANE, {SgprDst}, {VgprSrc});
    return;
  }

  SmallVector<Register, 8> SgprDstParts;
  unmergeReadAnyLane(B, SgprDstParts, getReadAnyLaneSplitTy(Ty), VgprSrc, RBI);

  B.buildMergeLikeInstr(SgprDst, SgprDstParts).getReg(0);
}
