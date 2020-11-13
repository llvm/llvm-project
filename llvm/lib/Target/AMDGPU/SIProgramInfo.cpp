//===-- SIProgramInfo.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// The SIProgramInfo tracks resource usage and hardware flags for kernels and
/// entry functions.
//
//===----------------------------------------------------------------------===//
//

#include "SIProgramInfo.h"
#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;

uint64_t SIProgramInfo::getComputePGMRSrc1() const {
  return S_00B848_VGPRS(VGPRBlocks) | S_00B848_SGPRS(SGPRBlocks) |
         S_00B848_PRIORITY(Priority) | S_00B848_FLOAT_MODE(FloatMode) |
         S_00B848_PRIV(Priv) | S_00B848_DX10_CLAMP(DX10Clamp) |
         S_00B848_DEBUG_MODE(DebugMode) | S_00B848_IEEE_MODE(IEEEMode) |
         S_00B848_WGP_MODE(WgpMode) | S_00B848_MEM_ORDERED(MemOrdered);
}

uint64_t SIProgramInfo::getPGMRSrc1(CallingConv::ID CC) const {
  if (AMDGPU::isCompute(CC)) {
    return getComputePGMRSrc1();
  }
  uint64_t Reg = S_00B848_VGPRS(VGPRBlocks) | S_00B848_SGPRS(SGPRBlocks) |
                 S_00B848_PRIORITY(Priority) | S_00B848_FLOAT_MODE(FloatMode) |
                 S_00B848_PRIV(Priv) | S_00B848_DX10_CLAMP(DX10Clamp) |
                 S_00B848_DEBUG_MODE(DebugMode) | S_00B848_IEEE_MODE(IEEEMode);
  switch (CC) {
  case CallingConv::AMDGPU_PS:
    Reg |= S_00B028_MEM_ORDERED(MemOrdered);
    break;
  case CallingConv::AMDGPU_VS:
    Reg |= S_00B128_MEM_ORDERED(MemOrdered);
    break;
  case CallingConv::AMDGPU_GS:
    Reg |= S_00B228_WGP_MODE(WgpMode) | S_00B228_MEM_ORDERED(MemOrdered);
    break;
  case CallingConv::AMDGPU_HS:
    Reg |= S_00B428_WGP_MODE(WgpMode) | S_00B428_MEM_ORDERED(MemOrdered);
    break;
  default:
    break;
  }
  return Reg;
}
