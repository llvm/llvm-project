//===-- M88kCallingConv.td - M88k Calling Conventions ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_M88KCALLINGCONV_H
#define LLVM_LIB_TARGET_M88K_M88KCALLINGCONV_H

#include "MCTargetDesc/M88kMCTargetDesc.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/Support/Debug.h"

namespace llvm {

inline bool CC_M88k_Custom_f64(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                               CCValAssign::LocInfo &LocInfo,
                               ISD::ArgFlagsTy &ArgFlags, CCState &State) {

  static const MCPhysReg HiRegList[] = {M88k::R2, M88k::R4, M88k::R6, M88k::R6};
  static const MCPhysReg LoRegList[] = {M88k::R3, M88k::R5, M88k::R7, M88k::R9};
  static const MCPhysReg ShadowRegList[] = {M88k::R1, M88k::R3, M88k::R5,
                                            M88k::R7};

  MCRegister RegHi = State.AllocateReg(HiRegList, ShadowRegList);
  if (RegHi == 0)
    return false; // TODO Do we need to allocate unused register?

  MCRegister RegLo = State.AllocateReg(LoRegList);
  assert(RegLo && "Could not allocate odd part of register pair");

  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, RegHi, LocVT, LocInfo));
  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, RegLo, LocVT, LocInfo));
  return true;
}

inline bool RetCC_M88k_Custom_f64(unsigned &ValNo, MVT &ValVT, MVT &LocVT,
                                  CCValAssign::LocInfo &LocInfo,
                                  ISD::ArgFlagsTy &ArgFlags, CCState &State) {
  MCRegister RegHi = State.AllocateReg(M88k::R2);
  if (RegHi == 0)
    return false;

  MCRegister RegLo = State.AllocateReg(M88k::R3);
  assert(RegLo && "Could not allocate odd part of register pair");

  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, RegHi, LocVT, LocInfo));
  State.addLoc(CCValAssign::getCustomReg(ValNo, ValVT, RegLo, LocVT, LocInfo));
  return true;
}
} // end namespace llvm

#endif
