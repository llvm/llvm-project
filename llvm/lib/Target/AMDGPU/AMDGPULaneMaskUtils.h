//===- AMDGPULaneMaskUtils.h - Exec/lane mask helper functions -*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H

#include "llvm/CodeGen/Register.h"

namespace llvm {

class GCNSubtarget;

namespace AMDGPU {

class LaneMaskConstants {
public:
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

  LaneMaskConstants(unsigned WavefrontSize);
};

const LaneMaskConstants &getLaneMaskConstants(const GCNSubtarget *ST);

} // end namespace AMDGPU

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_UTILS_AMDGPULANEMASKUTILS_H
