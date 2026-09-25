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

static const AMDGPU::VLdStIdxOpcodeInfo &getInfo(unsigned Opcode) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByOpcode(Opcode);
  if (!Info)
    llvm_unreachable("unsupported V_LOAD/STORE_IDX opcode");
  return *Info;
}

bool VLoadStoreIdxInst::isGPRIdx() const {
  return getInfo(getOpcode()).IsGPRIdx;
}

unsigned VLoadStoreIdxInst::getBitWidth() const {
  return getInfo(getOpcode()).BitWidth;
}

int VLoadIdxInst::tryGetOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByKey(Bits, /*IsStore=*/false, IsGPRIdx);
  if (!Info)
    return -1;
  return Info->Opcode;
}

unsigned VLoadIdxInst::getOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx) {
  int Opcode = tryGetOpcodeForBitWidth(Bits, IsGPRIdx);
  assert(Opcode != -1);
  return Opcode;
}

int VStoreIdxInst::tryGetOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx) {
  const AMDGPU::VLdStIdxOpcodeInfo *Info =
      AMDGPU::getVLdStIdxOpcodeInfoByKey(Bits, /*IsStore=*/true, IsGPRIdx);
  if (!Info)
    return -1;
  return Info->Opcode;
}

unsigned VStoreIdxInst::getOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx) {
  int Opcode = tryGetOpcodeForBitWidth(Bits, IsGPRIdx);
  assert(Opcode != -1);
  return Opcode;
}
