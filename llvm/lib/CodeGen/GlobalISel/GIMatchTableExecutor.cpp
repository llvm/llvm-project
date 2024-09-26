//===- llvm/CodeGen/GlobalISel/GIMatchTableExecutor.cpp -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file implements the GIMatchTableExecutor class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "gi-match-table-executor"

using namespace llvm;

GIMatchTableExecutor::MatcherState::MatcherState(unsigned MaxRenderers)
    : Renderers(MaxRenderers) {}

GIMatchTableExecutor::GIMatchTableExecutor() = default;

bool GIMatchTableExecutor::isOperandImmEqual(const MachineOperand &MO,
                                             int64_t Value,
                                             const MachineRegisterInfo &MRI,
                                             bool Splat) const {
  if (MO.isReg() && MO.getReg()) {
    if (auto VRegVal = getIConstantVRegValWithLookThrough(MO.getReg(), MRI))
      return VRegVal->Value.getSExtValue() == Value;

    if (Splat) {
      if (auto VRegVal = getIConstantSplatVal(MO.getReg(), MRI))
        return VRegVal->getSExtValue() == Value;
    }
  }
  return false;
}

bool GIMatchTableExecutor::isBaseWithConstantOffset(
    const MachineOperand &Root, const MachineRegisterInfo &MRI) const {
  if (!Root.isReg())
    return false;

  MachineInstr *RootI = MRI.getVRegDef(Root.getReg());
  if (RootI->getOpcode() != TargetOpcode::G_PTR_ADD)
    return false;

  MachineOperand &RHS = RootI->getOperand(2);
  MachineInstr *RHSI = MRI.getVRegDef(RHS.getReg());
  if (RHSI->getOpcode() != TargetOpcode::G_CONSTANT)
    return false;

  return true;
}

bool GIMatchTableExecutor::isObviouslySafeToFold(MachineInstr &MI,
                                                 MachineInstr &IntoMI) const {
  auto IntoMIIter = IntoMI.getIterator();

  // Immediate neighbours are already folded.
  if (MI.getParent() == IntoMI.getParent() &&
      std::next(MI.getIterator()) == IntoMIIter)
    return true;

  // Convergent instructions cannot be moved in the CFG.
  if (MI.isConvergent() && MI.getParent() != IntoMI.getParent())
    return false;

  if (MI.isLoadFoldBarrier())
    return false;

  // If the load is simple, check instructions between MI and IntoMI
  if (MI.mayLoad() && MI.getParent() == IntoMI.getParent()) {
    if (MI.memoperands_empty())
      return false;
    auto &MMO = **(MI.memoperands_begin());
    if (MMO.isAtomic() || MMO.isVolatile())
      return false;

    // Ensure instructions between MI and IntoMI are not affected when combined
    unsigned Iter = 0;
    const unsigned MaxIter = 20;
    for (auto &CurrMI :
         instructionsWithoutDebug(MI.getIterator(), IntoMI.getIterator())) {
      if (CurrMI.isLoadFoldBarrier())
        return false;

      if (Iter++ == MaxIter)
        return false;
    }

    return true;
  }

  return true;
}
