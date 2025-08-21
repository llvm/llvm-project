//===-- AMDGPULaneMaskUtils.cpp - -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPULaneMaskUtils.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"

#define DEBUG_TYPE "amdgpu-lane-mask-utils"

using namespace llvm;

namespace llvm::AMDGPU {

LaneMaskConstants::LaneMaskConstants(unsigned WavefrontSize) {
  if (WavefrontSize == 32) {
    ExecReg = AMDGPU::EXEC_LO;
    VccReg = AMDGPU::VCC_LO;
    AndOpc = AMDGPU::S_AND_B32;
    AndTermOpc = AMDGPU::S_AND_B32_term;
    AndN2Opc = AMDGPU::S_ANDN2_B32;
    AndN2SaveExecOpc = AMDGPU::S_ANDN2_SAVEEXEC_B32;
    AndN2TermOpc = AMDGPU::S_ANDN2_B32_term;
    AndSaveExecOpc = AMDGPU::S_AND_SAVEEXEC_B32;
    AndSaveExecTermOpc = AMDGPU::S_AND_SAVEEXEC_B32_term;
    BfmOpc = AMDGPU::S_BFM_B32;
    CMovOpc = AMDGPU::S_CMOV_B32;
    CSelectOpc = AMDGPU::S_CSELECT_B32;
    MovOpc = AMDGPU::S_MOV_B32;
    MovTermOpc = AMDGPU::S_MOV_B32_term;
    OrOpc = AMDGPU::S_OR_B32;
    OrTermOpc = AMDGPU::S_OR_B32_term;
    OrSaveExecOpc = AMDGPU::S_OR_SAVEEXEC_B32;
    XorOpc = AMDGPU::S_XOR_B32;
    XorTermOpc = AMDGPU::S_XOR_B32_term;
    WQMOpc = AMDGPU::S_WQM_B32;
  } else {
    ExecReg = AMDGPU::EXEC;
    VccReg = AMDGPU::VCC;
    AndOpc = AMDGPU::S_AND_B64;
    AndTermOpc = AMDGPU::S_AND_B64_term;
    AndN2Opc = AMDGPU::S_ANDN2_B64;
    AndN2SaveExecOpc = AMDGPU::S_ANDN2_SAVEEXEC_B64;
    AndN2TermOpc = AMDGPU::S_ANDN2_B64_term;
    AndSaveExecOpc = AMDGPU::S_AND_SAVEEXEC_B64;
    AndSaveExecTermOpc = AMDGPU::S_AND_SAVEEXEC_B64_term;
    BfmOpc = AMDGPU::S_BFM_B64;
    CMovOpc = AMDGPU::S_CMOV_B64;
    CSelectOpc = AMDGPU::S_CSELECT_B64;
    MovOpc = AMDGPU::S_MOV_B64;
    MovTermOpc = AMDGPU::S_MOV_B64_term;
    OrOpc = AMDGPU::S_OR_B64;
    OrTermOpc = AMDGPU::S_OR_B64_term;
    OrSaveExecOpc = AMDGPU::S_OR_SAVEEXEC_B64;
    XorOpc = AMDGPU::S_XOR_B64;
    XorTermOpc = AMDGPU::S_XOR_B64_term;
    WQMOpc = AMDGPU::S_WQM_B64;
  }
}

static const LaneMaskConstants Wave32LaneMaskConstants(32);
static const LaneMaskConstants Wave64LaneMaskConstants(64);

const LaneMaskConstants &getLaneMaskConstants(const GCNSubtarget *ST) {
  unsigned WavefrontSize = ST->getWavefrontSize();
  assert(WavefrontSize == 32 || WavefrontSize == 64);
  return WavefrontSize == 32 ? Wave32LaneMaskConstants
                             : Wave64LaneMaskConstants;
}

} // end namespace llvm::AMDGPU
