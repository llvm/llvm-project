//===- AMDGPUGlobalISelUtils -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/Register.h"
#include <utility>

namespace llvm {

class MachineRegisterInfo;
class GCNSubtarget;
class GISelValueTracking;
class LLT;
class MachineFunction;
class MachineIRBuilder;
class RegisterBankInfo;

namespace AMDGPU {

/// Returns base register and constant offset.
std::pair<Register, unsigned>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg,
                          GISelValueTracking *ValueTracking = nullptr,
                          bool CheckNUW = false);

// Finds lane masks produced/consumed by the control flow intrinsics. These are
// i1 values that live in wave-width lane mask registers (SReg_1). They are used
// to assign such values to the Vcc (lane mask) register bank so that they
// select to a wave mask register class.
class IntrinsicLaneMaskAnalyzer {
  SmallDenseSet<Register, 8> LaneMask;
  MachineRegisterInfo &MRI;

public:
  IntrinsicLaneMaskAnalyzer(MachineFunction &MF);
  bool isLaneMask(Register Reg) const;

private:
  void initLaneMaskIntrinsics(MachineFunction &MF);
};

void buildReadAnyLane(MachineIRBuilder &B, Register SgprDst, Register VgprSrc,
                      const RegisterBankInfo &RBI);
void buildReadFirstLane(MachineIRBuilder &B, Register SgprDst, Register VgprSrc,
                        const RegisterBankInfo &RBI);
}
}

#endif
