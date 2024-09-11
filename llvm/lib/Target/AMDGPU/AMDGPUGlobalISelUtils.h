//===- AMDGPUGlobalISelUtils -------------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUGLOBALISELUTILS_H

#include "AMDGPURegisterBankInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/Register.h"
#include <utility>

namespace llvm {

class MachineRegisterInfo;
class GCNSubtarget;
class GISelKnownBits;
class LLT;

namespace AMDGPU {

/// Returns base register and constant offset.
std::pair<Register, unsigned>
getBaseWithConstantOffset(MachineRegisterInfo &MRI, Register Reg,
                          GISelKnownBits *KnownBits = nullptr,
                          bool CheckNUW = false);

// Currently finds S32/S64 lane masks that can be declared as divergent by
// uniformity analysis (all are phis at the moment).
// These are defined as i32/i64 in some IR intrinsics (not as i1).
// Tablegen forces(via telling that lane mask IR intrinsics are uniform) most of
// S32/S64 lane masks to be uniform, as this results in them ending up with sgpr
// reg class after instruction-select don't search for all of them.
class IntrinsicLaneMaskAnalyzer {
  DenseSet<Register> S32S64LaneMask;
  MachineRegisterInfo &MRI;

public:
  IntrinsicLaneMaskAnalyzer(MachineFunction &MF);
  bool isS32S64LaneMask(Register Reg);

private:
  void initLaneMaskIntrinsics(MachineFunction &MF);
  // This will not be needed when we turn of LCSSA for global-isel.
  void findLCSSAPhi(Register Reg);
};

void buildReadAnyLaneS1(MachineIRBuilder &B, MachineInstr &MI,
                        const RegisterBankInfo &RBI);

MachineInstrBuilder buildReadAnyLaneB32(MachineIRBuilder &B,
                                        const DstOp &SgprDst,
                                        const SrcOp &VgprSrc,
                                        const RegisterBankInfo &RBI);

MachineInstrBuilder buildReadAnyLaneSequenceOfB32(MachineIRBuilder &B,
                                                  const DstOp &SgprDst,
                                                  const SrcOp &VgprSrc,
                                                  LLT B32Ty,
                                                  const RegisterBankInfo &RBI);

MachineInstrBuilder buildReadAnyLaneSequenceOfS64(MachineIRBuilder &B,
                                                  const DstOp &SgprDst,
                                                  const SrcOp &VgprSrc,
                                                  const RegisterBankInfo &RBI);

MachineInstrBuilder buildReadAnyLane(MachineIRBuilder &B, const DstOp &SgprDst,
                                     const SrcOp &VgprSrc,
                                     const RegisterBankInfo &RBI);

// Create new vgpr destination register for MI then move it to current
// MI's sgpr destination using one or more G_READANYLANE instructions.
void buildReadAnyLaneDst(MachineIRBuilder &B, MachineInstr &MI,
                         const RegisterBankInfo &RBI);

// Share with SIRegisterInfo::isUniformReg? This could make uniformity info give
// same result in later passes.
bool isLaneMask(Register Reg, MachineRegisterInfo &MRI,
                const SIRegisterInfo *TRI);

bool isSgprRB(Register Reg, MachineRegisterInfo &MRI);

bool isVgprRB(Register Reg, MachineRegisterInfo &MRI);

template <typename SrcTy>
inline MIPatternMatch::UnaryOp_match<SrcTy, AMDGPU::G_READANYLANE>
m_GReadAnyLane(const SrcTy &Src) {
  return MIPatternMatch::UnaryOp_match<SrcTy, AMDGPU::G_READANYLANE>(Src);
}

void cleanUpAfterCombine(MachineInstr &MI, MachineRegisterInfo &MRI,
                         MachineInstr *Optional0 = nullptr);

bool hasSGPRS1(MachineFunction &MF, MachineRegisterInfo &MRI);

bool isS1(Register Reg, MachineRegisterInfo &MRI);

} // namespace AMDGPU
} // namespace llvm

#endif
