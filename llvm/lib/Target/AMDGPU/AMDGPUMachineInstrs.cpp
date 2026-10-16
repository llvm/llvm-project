//===-- AMDGPUMachineInstrs.cpp -*- C++ -*---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Convenience wrappers and helpers for AMDGPU-specific machine instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMachineInstrs.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;
using namespace AMDGPUMI;

unsigned VLoadStoreIdxInst::getNumIndexableVGPRs(const MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  unsigned DynamicVGPRBlockSize =
      MF.getInfo<SIMachineFunctionInfo>()->getDynamicVGPRBlockSize();
  return std::min(ST.getAddressableNumArchVGPRs(),
                  ST.getAddressableNumVGPRs(DynamicVGPRBlockSize));
}

int VLoadIdxInst::tryGetOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByKey(Bits, /*IsStore=*/false, IsGPRIdx);
  if (!Info)
    return -1;
  return Info->Opcode;
}

int VStoreIdxInst::tryGetOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByKey(Bits, /*IsStore=*/true, IsGPRIdx);
  if (!Info)
    return -1;
  return Info->Opcode;
}
