//===- MachineFloatingPointPredicateUtils.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/MachineFloatingPointPredicateUtils.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/IR/Constants.h"
#include <optional>

namespace llvm {

using namespace MIPatternMatch;

template <>
DenormalMode
MachineFloatingPointPredicateUtils::queryDenormalMode(const MachineFunction &MF,
                                                      Register Val) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  LLT Ty = MRI.getType(Val).getScalarType();
  return MF.getDenormalMode(getFltSemanticForLLT(Ty));
}

template <>
bool MachineFloatingPointPredicateUtils::lookThroughFAbs(
    const MachineFunction &MF, Register LHS, Register &Src) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  return mi_match(LHS, MRI, m_GFabs(m_Reg(Src)));
}

template <>
std::optional<APFloat> MachineFloatingPointPredicateUtils::matchConstantFloat(
    const MachineFunction &MF, Register Val) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const ConstantFP *ConstVal;
  if (mi_match(Val, MRI, m_GFCst(ConstVal)))
    return ConstVal->getValueAPF();

  return std::nullopt;
}

} // namespace llvm
