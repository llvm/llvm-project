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
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace AMDGPUMI;

unsigned VLoadStoreIdxInst::getBitWidth() const {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByOpcode(getOpcode());
  if (!Info)
    llvm_unreachable("unsupported V_LOAD/STORE_IDX opcode");
  return Info->BitWidth;
}

int VLoadIdxInst::tryGetOpcodeForBitWidth(unsigned Bits) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByKey(Bits, /*IsStore=*/false);
  if (!Info)
    return -1;
  return Info->Opcode;
}

unsigned VLoadIdxInst::getOpcodeForBitWidth(unsigned Bits) {
  int Opcode = tryGetOpcodeForBitWidth(Bits);
  assert(Opcode != -1);
  return Opcode;
}

int VStoreIdxInst::tryGetOpcodeForBitWidth(unsigned Bits) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByKey(Bits, /*IsStore=*/true);
  if (!Info)
    return -1;
  return Info->Opcode;
}

unsigned VStoreIdxInst::getOpcodeForBitWidth(unsigned Bits) {
  int Opcode = tryGetOpcodeForBitWidth(Bits);
  assert(Opcode != -1);
  return Opcode;
}
