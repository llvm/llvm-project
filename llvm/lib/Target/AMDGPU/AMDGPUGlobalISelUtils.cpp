//===- AMDGPUGlobalISelUtils.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUGlobalISelUtils.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/LowLevelTypeImpl.h"

using namespace llvm;
using namespace MIPatternMatch;

std::pair<Register, unsigned>
AMDGPU::getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg,
                                  GISelKnownBits *KnownBits) {
  MachineInstr *Def = getDefIgnoringCopies(Reg, MRI);
  if (!Def)
    return std::make_pair(Reg, 0);

  if (Def->getOpcode() == TargetOpcode::G_CONSTANT) {
    unsigned Offset;
    const MachineOperand &Op = Def->getOperand(1);
    if (Op.isImm())
      Offset = Op.getImm();
    else
      Offset = Op.getCImm()->getZExtValue();

    return std::make_pair(Register(), Offset);
  }

  int64_t Offset;
  if (Def->getOpcode() == TargetOpcode::G_ADD) {
    // TODO: Handle G_OR used for add case
    if (mi_match(Def->getOperand(2).getReg(), MRI, m_ICst(Offset)))
      return std::make_pair(Def->getOperand(1).getReg(), Offset);

    // FIXME: matcher should ignore copies
    if (mi_match(Def->getOperand(2).getReg(), MRI, m_Copy(m_ICst(Offset))))
      return std::make_pair(Def->getOperand(1).getReg(), Offset);
  }

  Register Base;
  if (KnownBits && mi_match(Reg, MRI, m_GOr(m_Reg(Base), m_ICst(Offset))) &&
      KnownBits->maskedValueIsZero(Base, APInt(32, Offset)))
    return std::make_pair(Base, Offset);

  // Handle G_PTRTOINT (G_PTR_ADD base, const) case
  if (Def->getOpcode() == TargetOpcode::G_PTRTOINT) {
    MachineInstr *Base;
    if (mi_match(Def->getOperand(1).getReg(), MRI,
                 m_GPtrAdd(m_MInstr(Base), m_ICst(Offset)))) {
      // If Base was int converted to pointer, simply return int and offset.
      if (Base->getOpcode() == TargetOpcode::G_INTTOPTR)
        return std::make_pair(Base->getOperand(1).getReg(), Offset);

      // Register returned here will be of pointer type.
      return std::make_pair(Base->getOperand(0).getReg(), Offset);
    }
  }

  return std::make_pair(Reg, 0);
}

bool AMDGPU::isLegalVOP3PShuffleMask(ArrayRef<int> Mask) {
  assert(Mask.size() == 2);

  // If one half is undef, the other is trivially in the same reg.
  if (Mask[0] == -1 || Mask[1] == -1)
    return true;
  return (Mask[0] & 2) == (Mask[1] & 2);
}

bool AMDGPU::hasAtomicFaddRtnForTy(const GCNSubtarget &Subtarget,
                                   const LLT &Ty) {
  if (Ty == LLT::scalar(32))
    return Subtarget.hasAtomicFaddRtnInsts();
  if (Ty == LLT::fixed_vector(2, 16) || Ty == LLT::scalar(64))
    return Subtarget.hasGFX90AInsts();
  return false;
}
