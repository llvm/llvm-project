//===- AMDGPULaneMaskUtils.h - Exec/lane mask helper functions -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H

#include "GCNSubtarget.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {

class GCNSubtarget;

namespace AMDGPU {

struct LaneMaskConstants {
  Register ExecReg;
  Register VccReg;
  unsigned AndOpc;
  unsigned AndTermOpc;
  unsigned AndN2Opc;
  unsigned AndN2SaveExecOpc;
  unsigned AndN2TermOpc;
  unsigned AndSaveExecOpc;
  unsigned AndSaveExecTermOpc;
  unsigned BfmOpc;
  unsigned CMovOpc;
  unsigned CSelectOpc;
  unsigned MovOpc;
  unsigned MovTermOpc;
  unsigned OrOpc;
  unsigned OrTermOpc;
  unsigned OrSaveExecOpc;
  unsigned XorOpc;
  unsigned XorTermOpc;
  unsigned WQMOpc;

  static constexpr LaneMaskConstants getWave32() {
    // clang-format off
    return {
      AMDGPU::EXEC_LO,
      AMDGPU::VCC_LO,
      AMDGPU::S_AND_B32,
      AMDGPU::S_AND_B32_term,
      AMDGPU::S_ANDN2_B32,
      AMDGPU::S_ANDN2_SAVEEXEC_B32,
      AMDGPU::S_ANDN2_B32_term,
      AMDGPU::S_AND_SAVEEXEC_B32,
      AMDGPU::S_AND_SAVEEXEC_B32_term,
      AMDGPU::S_BFM_B32,
      AMDGPU::S_CMOV_B32,
      AMDGPU::S_CSELECT_B32,
      AMDGPU::S_MOV_B32,
      AMDGPU::S_MOV_B32_term,
      AMDGPU::S_OR_B32,
      AMDGPU::S_OR_B32_term,
      AMDGPU::S_OR_SAVEEXEC_B32,
      AMDGPU::S_XOR_B32,
      AMDGPU::S_XOR_B32_term,
      AMDGPU::S_WQM_B32
    };
    // clang-format on
  }

  static constexpr LaneMaskConstants getWave64() {
    // clang-format off
    return {
      AMDGPU::EXEC,
      AMDGPU::VCC,
      AMDGPU::S_AND_B64,
      AMDGPU::S_AND_B64_term,
      AMDGPU::S_ANDN2_B64,
      AMDGPU::S_ANDN2_SAVEEXEC_B64,
      AMDGPU::S_ANDN2_B64_term,
      AMDGPU::S_AND_SAVEEXEC_B64,
      AMDGPU::S_AND_SAVEEXEC_B64_term,
      AMDGPU::S_BFM_B64,
      AMDGPU::S_CMOV_B64,
      AMDGPU::S_CSELECT_B64,
      AMDGPU::S_MOV_B64,
      AMDGPU::S_MOV_B64_term,
      AMDGPU::S_OR_B64,
      AMDGPU::S_OR_B64_term,
      AMDGPU::S_OR_SAVEEXEC_B64,
      AMDGPU::S_XOR_B64,
      AMDGPU::S_XOR_B64_term,
      AMDGPU::S_WQM_B64
    };
    // clang-format on
  }

  static LaneMaskConstants get(const GCNSubtarget *ST) {
    unsigned WavefrontSize = ST->getWavefrontSize();
    assert(WavefrontSize == 32 || WavefrontSize == 64);
    return WavefrontSize == 32 ? LaneMaskConstants::getWave32()
                               : LaneMaskConstants::getWave64();
  }
};

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H
