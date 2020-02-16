//===- AMDGPUGlobalISelUtils.cpp ---------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUGlobalISelUtils.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/IR/Constants.h"

using namespace llvm;
using namespace MIPatternMatch;

std::tuple<Register, unsigned, MachineInstr *>
AMDGPU::getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg) {
  MachineInstr *Def = getDefIgnoringCopies(Reg, MRI);
  if (!Def)
    return std::make_tuple(Reg, 0, nullptr);

  if (Def->getOpcode() == TargetOpcode::G_CONSTANT) {
    unsigned Offset;
    const MachineOperand &Op = Def->getOperand(1);
    if (Op.isImm())
      Offset = Op.getImm();
    else
      Offset = Op.getCImm()->getZExtValue();

    return std::make_tuple(Register(), Offset, Def);
  }

  int64_t Offset;
  if (Def->getOpcode() == TargetOpcode::G_ADD) {
    // TODO: Handle G_OR used for add case
    if (mi_match(Def->getOperand(2).getReg(), MRI, m_ICst(Offset)))
      return std::make_tuple(Def->getOperand(1).getReg(), Offset, Def);

    // FIXME: matcher should ignore copies
    if (mi_match(Def->getOperand(2).getReg(), MRI, m_Copy(m_ICst(Offset))))
      return std::make_tuple(Def->getOperand(1).getReg(), Offset, Def);
  }

  return std::make_tuple(Reg, 0, Def);
}

bool AMDGPU::isLegalVOP3PShuffleMask(ArrayRef<int> Mask) {
  assert(Mask.size() == 2);

  // If one half is undef, the other is trivially in the same reg.
  if (Mask[0] == -1 || Mask[1] == -1)
    return true;
  return (Mask[0] & 2) == (Mask[1] & 2);
}
