//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains helper functions to find and list registers that are
/// tracked by the unwinding information checker.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFCFICHECKER_REGISTERS_H
#define LLVM_DWARFCFICHECKER_REGISTERS_H

#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {

/// This analysis only keeps track and cares about super registers, not the
/// subregisters. All reads from/writes to subregisters are considered the
/// same operation to super registers.
inline bool isSuperReg(const MCRegisterInfo *MCRI, MCRegister Reg) {
  return MCRI->superregs(Reg).empty();
}

inline SmallVector<MCPhysReg> getSuperRegs(const MCRegisterInfo *MCRI) {
  SmallVector<MCPhysReg> SuperRegs;
  for (auto &&RegClass : MCRI->regclasses())
    for (unsigned I = 0; I < RegClass.getNumRegs(); I++) {
      MCRegister Reg = RegClass.getRegister(I);
      if (isSuperReg(MCRI, Reg))
        SuperRegs.push_back(Reg.id());
    }

  sort(SuperRegs.begin(), SuperRegs.end());
  SuperRegs.erase(llvm::unique(SuperRegs), SuperRegs.end());
  return SuperRegs;
}

inline SmallVector<MCPhysReg> getTrackingRegs(const MCRegisterInfo *MCRI) {
  SmallVector<MCPhysReg> TrackingRegs;
  for (auto Reg : getSuperRegs(MCRI))
    if (!MCRI->isArtificial(Reg) && !MCRI->isConstant(Reg))
      TrackingRegs.push_back(Reg);
  return TrackingRegs;
}

inline MCRegister getSuperReg(const MCRegisterInfo *MCRI, MCRegister Reg) {
  if (isSuperReg(MCRI, Reg))
    return Reg;
  for (auto SuperReg : MCRI->superregs(Reg))
    if (isSuperReg(MCRI, SuperReg))
      return SuperReg;

  llvm_unreachable("Should either be a super reg, or have a super reg");
}

} // namespace llvm

#endif
